package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
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
			break
		case "assistant":
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeReasoning:
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

					if part.State != ToolInvocationStateOutputAvailable && part.State != ToolInvocationStateOutputError {
						continue
					}

					// Tool Results are sent as a separate message, so we need to flush existing content here.
					anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
						Role:    role,
						Content: content,
					})
					content = nil

					resultContent := []anthropic.ToolResultBlockParamContentUnion{}

					if part.State == ToolInvocationStateOutputError {
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

					// Send the tool result as a separate message with the role as user.
					anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
						Role: anthropic.MessageParamRoleUser,
						Content: []anthropic.ContentBlockParamUnion{
							{
								OfToolResult: &anthropic.ToolResultBlockParam{
									ToolUseID: part.ToolCallID,
									Content:   resultContent,
								},
							},
						},
					})
					content = nil
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
			content = nil
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
		var finalReason FinishReason = FinishReasonUnknown
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
					if currentToolCall.Args != "" {
						if err := json.Unmarshal([]byte(currentToolCall.Args), &input); err != nil {
							yield(nil, fmt.Errorf("unmarshalling tool input for call %s %q: %w", currentToolCall.ID, currentToolCall.Args, err))
							return
						}
					}

					if !yield(ToolInputAvailablePart{
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

			case anthropic.MessageStopEvent:
				// Determine final reason if not already set by tool_use
				if finalReason == FinishReasonUnknown {
					finalReason = FinishReasonStop // Default if not tool_use
				}

				// Send final finish step
				if !yield(FinishStepPart{}, nil) {
					return
				}

				// Send final finish message
				if !yield(FinishPart{}, nil) {
					return
				}
			}
		}

		// Handle any errors from the stream
		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("anthropic stream error: %w", err))
			return
		}

		// If we didn't get a message stop event (e.g., stream ended abruptly),
		// send a final finish message based on the last known state.
		if lastChunk == nil || lastChunk.Type != "message_stop" {
			if finalReason == FinishReasonUnknown {
				finalReason = FinishReasonError // Indicate abnormal termination
			}

			yield(FinishStepPart{}, nil)

			yield(FinishPart{
				// FinishReason: finalReason,
				// Usage:        finalUsage,
			}, nil)
		}
	}

	return dataStream, getUsage
}
