package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

// ToolsToAnthropic converts the tool format to Anthropic's API format.
func ToolsToAnthropic(tools []Tool) []anthropic.ToolUnionParam {
	anthropicTools := []anthropic.ToolUnionParam{}
	for _, tool := range tools {
		// Construct the ToolInputSchemaParam struct directly
		properties := tool.Schema.Properties
		if properties == nil {
			properties = make(map[string]interface{})
		}
		inputSchema := anthropic.ToolInputSchemaParam{
			Properties: properties, // Assuming Properties is map[string]interface{}
			// Type defaults to "object" via omitempty / SDK marshalling if needed
		}
		// Add required fields if they exist
		if len(tool.Schema.Required) > 0 {
			if inputSchema.ExtraFields == nil {
				inputSchema.ExtraFields = make(map[string]interface{})
			}
			inputSchema.ExtraFields["required"] = tool.Schema.Required
		}

		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Description),
				InputSchema: inputSchema, // Assign the struct directly
			},
		})
	}
	return anthropicTools
}

// MessagesToAnthropic converts internal message format to Anthropic's API format.
// It extracts system messages into a separate slice of TextBlockParams and groups
// consecutive user/tool and assistant messages according to Anthropic's rules.
// It handles the case where a single assistant message part contains both the
// tool call and its result, splitting them into the required assistant tool_use
// and user tool_result blocks.
func MessagesToAnthropic(messages []Message) ([]anthropic.MessageParam, []anthropic.TextBlockParam, error) {
	anthropicMessages := []anthropic.MessageParam{}

	var systemPrompt []anthropic.TextBlockParam

	for _, message := range messages {
		role := anthropic.MessageParamRoleAssistant
		content := []anthropic.ContentBlockParamUnion{}

		switch message.Role {
		case "system":
			if len(systemPrompt) > 0 {
				return nil, nil, fmt.Errorf("multiple system messages found")
			}
			for _, part := range message.Parts {
				if part.Type == PartTypeText && part.Text != "" {
					systemPrompt = append(systemPrompt, anthropic.TextBlockParam{
						Text: part.Text,
					})
				}
			}
		case "assistant":
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeReasoning:
					if part.ProviderMetadata != nil && part.ProviderMetadata.Anthropic != nil && part.ProviderMetadata.Anthropic.RedactedData != "" {
						content = append(content, anthropic.ContentBlockParamUnion{
							OfRedactedThinking: &anthropic.RedactedThinkingBlockParam{
								Data: part.ProviderMetadata.Anthropic.RedactedData,
							},
						})
						continue
					}
					content = append(content, anthropic.ContentBlockParamUnion{
						OfThinking: &anthropic.ThinkingBlockParam{
							Thinking:  part.Text,
							Signature: part.ProviderMetadata.Anthropic.Signature,
						},
					})
				case PartTypeText:
					content = append(content, anthropic.ContentBlockParamUnion{
						OfText: &anthropic.TextBlockParam{
							Text: part.Text,
						},
					})
				case PartTypeToolInvocation:
					if part.Input == nil {
						part.Input = make(map[string]any)
					}
					argsJSON, err := json.Marshal(part.Input)
					if err != nil {
						return nil, nil, fmt.Errorf("marshalling tool input for call %s: %w", part.ToolCallID, err)
					}
					content = append(content, anthropic.ContentBlockParamUnion{
						OfToolUse: &anthropic.ToolUseBlockParam{
							ID:    part.ToolCallID,
							Input: json.RawMessage(argsJSON),
							Name:  part.ToolName,
						},
					})

					denied := isDeniedToolPart(part)
					if part.State != ToolStateOutputAvailable && part.State != ToolStateOutputError && !denied {
						continue
					}

					// Tool Results are sent as a separate message, so we need to flush existing content here.
					anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
						Role:    role,
						Content: content,
					})
					content = nil

					resultContent := []anthropic.ToolResultBlockParamContentUnion{}

					if denied {
						resultContent = append(resultContent, anthropic.ToolResultBlockParamContentUnion{
							OfText: &anthropic.TextBlockParam{Text: deniedToolResultReason(part)},
						})
					} else if part.State == ToolStateOutputError {
						resultContent = append(resultContent, anthropic.ToolResultBlockParamContentUnion{
							OfText: &anthropic.TextBlockParam{Text: part.ErrorText},
						})
					} else {
						resultParts, err := toolResultToParts(part.Output)
						if err != nil {
							return nil, nil, fmt.Errorf("failed to convert tool call result to parts: %w", err)
						}
						for _, resultPart := range resultParts {
							switch resultPart.Type {
							case PartTypeText:
								resultContent = append(resultContent, anthropic.ToolResultBlockParamContentUnion{
									OfText: &anthropic.TextBlockParam{Text: resultPart.Text},
								})
							case PartTypeFile:
								resultContent = append(resultContent, anthropic.ToolResultBlockParamContentUnion{
									OfImage: &anthropic.ImageBlockParam{
										Source: anthropic.ImageBlockParamSourceUnion{
											OfBase64: &anthropic.Base64ImageSourceParam{
												Data:      base64.StdEncoding.EncodeToString(resultPart.Data),
												MediaType: anthropic.Base64ImageSourceMediaType(resultPart.MimeType),
											},
										},
									},
								})
							}
						}
					}

					toolResult := &anthropic.ToolResultBlockParam{
						ToolUseID: part.ToolCallID,
						Content:   resultContent,
					}
					// Mirrors the Vercel SDK: only the terminal output-denied
					// state is flagged as an error, a fresh denial is not.
					if part.State == ToolStateOutputDenied {
						toolResult.IsError = anthropic.Bool(true)
					}

					// Send the tool result as a separate message with the role as user.
					anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
						Role: anthropic.MessageParamRoleUser,
						Content: []anthropic.ContentBlockParamUnion{
							{OfToolResult: toolResult},
						},
					})
				}
			}
		case "user":
			role = anthropic.MessageParamRoleUser
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content = append(content, anthropic.ContentBlockParamUnion{
						OfText: &anthropic.TextBlockParam{Text: part.Text},
					})
				case PartTypeFile:
					content = append(content, anthropic.ContentBlockParamUnion{
						OfImage: &anthropic.ImageBlockParam{
							Source: anthropic.ImageBlockParamSourceUnion{
								OfBase64: &anthropic.Base64ImageSourceParam{
									Data:      base64.StdEncoding.EncodeToString(part.Data),
									MediaType: anthropic.Base64ImageSourceMediaType(part.MimeType),
								},
							},
						},
					})
				case PartTypeToolInvocation:
					return nil, nil, fmt.Errorf("user message part has type tool-invocation (ID: %s)", message.ID)
				}
			}
		default:
			return nil, nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}

		if len(message.Attachments) > 0 {
			for _, attachment := range message.Attachments {
				// URLs typically have the mime prefixing as a URL.
				parts := strings.SplitN(attachment.URL, ",", 2)
				if len(parts) != 2 {
					return nil, nil, fmt.Errorf("invalid attachment URL: %s", attachment.URL)
				}
				content = append(content, anthropic.ContentBlockParamUnion{
					OfImage: &anthropic.ImageBlockParam{
						Source: anthropic.ImageBlockParamSourceUnion{
							OfBase64: &anthropic.Base64ImageSourceParam{
								Data:      parts[1],
								MediaType: anthropic.Base64ImageSourceMediaType(attachment.ContentType),
							},
						},
					},
				})
			}
		}
		if len(content) > 0 {
			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role:    role,
				Content: content,
			})
		}
	}

	return anthropicMessages, systemPrompt, nil
}

// AnthropicToDataStream pipes an Anthropic stream to a DataStream.
func AnthropicToDataStream(stream *ssestream.Stream[anthropic.MessageStreamEventUnion]) (DataStream, func() anthropic.Usage) {
	usage := anthropic.Usage{}

	getUsage := func() anthropic.Usage {
		return usage
	}

	dataStream := func(yield func(DataStreamPart, error) bool) {
		var lastChunk *anthropic.MessageStreamEventUnion
		finalReason := FinishReasonUnknown
		// var finalUsage Usage
		var currentToolCall struct {
			ID   string
			Name string
			Args string
		}

		currentContentBlockType := ""
		currentContentBlockID := 0
		currentContentBlockIDText := "0"
		bumpContentBlockID := func() {
			currentContentBlockID++
			currentContentBlockIDText = fmt.Sprintf("%d", currentContentBlockID)
		}

		for stream.Next() {
			chunk := stream.Current()
			lastChunk = &chunk

			event := chunk.AsAny()
			switch event := event.(type) {
			case anthropic.MessageStartEvent:
				usage = event.Message.Usage

				if !yield(MessageStartPart{}, nil) {
					return
				}

				if !yield(StartStepStreamPart{}, nil) {
					return
				}

			case anthropic.ContentBlockDeltaEvent:
				switch delta := event.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					if !yield(TextDeltaPart{
						ID:    currentContentBlockIDText,
						Delta: delta.Text,
					}, nil) {
						return
					}
				case anthropic.SignatureDelta:
					if !yield(ReasoningDeltaPart{
						ID: currentContentBlockIDText,
						ProviderMetadata: ProviderMetadata{
							Anthropic: &AnthropicProviderMetadata{
								Signature: delta.Signature,
							},
						},
					}, nil) {
						return
					}
				case anthropic.ThinkingDelta:
					if !yield(ReasoningDeltaPart{
						ID:    currentContentBlockIDText,
						Delta: delta.Thinking,
					}, nil) {
						return
					}
				case anthropic.InputJSONDelta:
					// Accumulate the arguments for the current tool call
					currentToolCall.Args += delta.PartialJSON
					if !yield(ToolInputDeltaPart{
						ToolCallID:     currentToolCall.ID,
						InputTextDelta: delta.PartialJSON,
					}, nil) {
						return
					}
				}

			case anthropic.ContentBlockStartEvent:
				switch block := event.ContentBlock.AsAny().(type) {
				case anthropic.ThinkingBlock:
					if !yield(ReasoningStartPart{
						ID: currentContentBlockIDText,
					}, nil) {
						return
					}
					currentContentBlockType = "thinking"
				case anthropic.RedactedThinkingBlock:
					// The encrypted payload arrives complete in content_block_start;
					// there is no delta type for redacted thinking.
					if !yield(ReasoningStartPart{
						ID: currentContentBlockIDText,
					}, nil) {
						return
					}
					if !yield(ReasoningDeltaPart{
						ID: currentContentBlockIDText,
						ProviderMetadata: ProviderMetadata{
							Anthropic: &AnthropicProviderMetadata{
								RedactedData: block.Data,
							},
						},
					}, nil) {
						return
					}
					currentContentBlockType = "thinking"
				case anthropic.TextBlock:
					if !yield(TextStartPart{
						ID: currentContentBlockIDText,
					}, nil) {
						return
					}
					currentContentBlockType = "text"
				case anthropic.ToolUseBlock:
					currentToolCall.ID = block.ID
					currentToolCall.Name = block.Name
					currentToolCall.Args = ""

					if !yield(ToolInputStartPart{
						ToolCallID: block.ID,
						ToolName:   block.Name,
					}, nil) {
						return
					}
					currentContentBlockType = "tool_use"
				}

			case anthropic.ContentBlockStopEvent:
				switch currentContentBlockType {
				case "thinking":
					if !yield(ReasoningEndPart{
						ID: currentContentBlockIDText,
					}, nil) {
						return
					}
					bumpContentBlockID()
				case "text":
					if !yield(TextEndPart{
						ID: currentContentBlockIDText,
					}, nil) {
						return
					}
					bumpContentBlockID()
				case "tool_use":
					var input map[string]any
					var inputErr error
					if currentToolCall.Args != "" {
						inputErr = json.Unmarshal([]byte(currentToolCall.Args), &input)
					}

					if inputErr != nil {
						// Mirror Vercel: malformed tool input (typically max_tokens
						// truncation) never errors the stream. Emit the error parts
						// and keep going; the finish reason arrives later in
						// message_delta.
						errorText := fmt.Sprintf("unmarshalling tool input for call %s: %s", currentToolCall.ID, inputErr)
						if !yield(ToolInputErrorPart{
							ToolCallID: currentToolCall.ID,
							ToolName:   currentToolCall.Name,
							Input:      currentToolCall.Args,
							ErrorText:  errorText,
						}, nil) {
							return
						}
						if !yield(ToolOutputErrorPart{
							ToolCallID: currentToolCall.ID,
							ErrorText:  errorText,
						}, nil) {
							return
						}
					} else if !yield(ToolInputAvailablePart{
						ToolCallID: currentToolCall.ID,
						ToolName:   currentToolCall.Name,
						Input:      input,
					}, nil) {
						return
					}

					// Reset current tool call after emitting the final delta
					currentToolCall = struct {
						ID   string
						Name string
						Args string
					}{}

				}

			case anthropic.MessageDeltaEvent:
				usage.OutputTokens += event.Usage.OutputTokens
				if event.Delta.StopReason != "" {
					finalReason = mapAnthropicStopReason(event.Delta.StopReason)
				}

			case anthropic.MessageStopEvent:
				// Fall back to "stop" when the provider never reported a stop reason.
				if finalReason == FinishReasonUnknown {
					finalReason = FinishReasonStop
				}

				// Send final finish step
				if !yield(FinishStepPart{}, nil) {
					return
				}

				// Send final finish message
				if !yield(FinishPart{FinishReason: finalReason}, nil) {
					return
				}
			}
		}

		// Handle any errors from the stream
		// AWS eventstream decoder propagates io.EOF on normal completion so we need to ignore that.
		if err := stream.Err(); err != nil && err != io.EOF {
			if !yield(nil, fmt.Errorf("anthropic stream error: %w", err)) {
				return
			}
			return
		}

		// If we didn't get a message stop event (e.g., stream ended abruptly),
		// send a final finish message based on the last known state.
		if lastChunk == nil || lastChunk.Type != "message_stop" {
			if !yield(FinishStepPart{}, nil) {
				return
			}

			// "unknown" is not part of the client's finish-reason enum, so omit
			// the field entirely when the provider never reported a reason.
			finishPart := FinishPart{}
			if finalReason != FinishReasonUnknown {
				finishPart.FinishReason = finalReason
			}
			if !yield(finishPart, nil) {
				return
			}
		}
	}

	return dataStream, getUsage
}

// mapAnthropicStopReason mirrors the reference implementation's
// map-anthropic-stop-reason.ts, except the default maps to "other" instead of
// "unknown": the client's finish-reason enum has no "unknown", so strict
// clients would reject the finish part. The SDK has no constant for
// model_context_window_exceeded, hence the raw string case.
func mapAnthropicStopReason(stopReason anthropic.StopReason) FinishReason {
	switch stopReason {
	case anthropic.StopReasonPauseTurn, anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return FinishReasonStop
	case anthropic.StopReasonRefusal:
		return FinishReasonContentFilter
	case anthropic.StopReasonToolUse:
		return FinishReasonToolCalls
	case anthropic.StopReasonMaxTokens, "model_context_window_exceeded":
		return FinishReasonLength
	default:
		return FinishReasonOther
	}
}
