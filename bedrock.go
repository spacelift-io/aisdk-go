package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	bedrockdocument "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// BedrockConverseToDataStream pipes a Bedrock Converse stream to a DataStream.
func BedrockConverseToDataStream(stream bedrockruntime.ConverseStreamOutputReader) (DataStream, func() *bedrocktypes.TokenUsage) {
	var usage *bedrocktypes.TokenUsage

	getUsage := func() *bedrocktypes.TokenUsage {
		return usage
	}

	dataStream := func(yield func(DataStreamPart, error) bool) {
		type toolCallState struct {
			ID   string
			Name string
			Args string
		}

		messageStarted := false
		finalReason := FinishReasonStop
		currentContentBlockType := make(map[string]string)
		currentToolCalls := make(map[string]*toolCallState)

		startMessage := func() bool {
			if messageStarted {
				return true
			}

			messageStarted = true
			if !yield(MessageStartPart{}, nil) {
				return false
			}
			if !yield(StartStepStreamPart{}, nil) {
				return false
			}

			return true
		}

		for event := range stream.Events() {
			switch event := event.(type) {
			case *bedrocktypes.ConverseStreamOutputMemberMessageStart:
				if !startMessage() {
					return
				}

			case *bedrocktypes.ConverseStreamOutputMemberContentBlockStart:
				if !startMessage() {
					return
				}

				contentBlockID := bedrockContentBlockID(event.Value.ContentBlockIndex)

				switch start := event.Value.Start.(type) {
				case *bedrocktypes.ContentBlockStartMemberToolUse:
					toolCallID := ""
					if start.Value.ToolUseId != nil {
						toolCallID = *start.Value.ToolUseId
					}
					if toolCallID == "" {
						toolCallID = fmt.Sprintf("tool_%s", contentBlockID)
					}

					toolName := ""
					if start.Value.Name != nil {
						toolName = *start.Value.Name
					}

					currentToolCalls[contentBlockID] = &toolCallState{
						ID:   toolCallID,
						Name: toolName,
					}
					currentContentBlockType[contentBlockID] = "tool_use"

					if !yield(ToolInputStartPart{
						ToolCallID: toolCallID,
						ToolName:   toolName,
					}, nil) {
						return
					}

				default:
					currentContentBlockType[contentBlockID] = "text"
					if !yield(TextStartPart{ID: contentBlockID}, nil) {
						return
					}
				}

			case *bedrocktypes.ConverseStreamOutputMemberContentBlockDelta:
				if !startMessage() {
					return
				}

				contentBlockID := bedrockContentBlockID(event.Value.ContentBlockIndex)

				switch delta := event.Value.Delta.(type) {
				case *bedrocktypes.ContentBlockDeltaMemberText:
					if currentContentBlockType[contentBlockID] == "reasoning" {
						if !yield(ReasoningEndPart{ID: contentBlockID}, nil) {
							return
						}
						currentContentBlockType[contentBlockID] = ""
					}

					if currentContentBlockType[contentBlockID] != "text" {
						currentContentBlockType[contentBlockID] = "text"
						if !yield(TextStartPart{ID: contentBlockID}, nil) {
							return
						}
					}

					if !yield(TextDeltaPart{
						ID:    contentBlockID,
						Delta: delta.Value,
					}, nil) {
						return
					}

				case *bedrocktypes.ContentBlockDeltaMemberReasoningContent:
					if currentContentBlockType[contentBlockID] == "text" {
						if !yield(TextEndPart{ID: contentBlockID}, nil) {
							return
						}
						currentContentBlockType[contentBlockID] = ""
					}

					if currentContentBlockType[contentBlockID] != "reasoning" {
						currentContentBlockType[contentBlockID] = "reasoning"
						if !yield(ReasoningStartPart{ID: contentBlockID}, nil) {
							return
						}
					}

					switch reasoningDelta := delta.Value.(type) {
					case *bedrocktypes.ReasoningContentBlockDeltaMemberText:
						if !yield(ReasoningDeltaPart{
							ID:    contentBlockID,
							Delta: reasoningDelta.Value,
						}, nil) {
							return
						}

					case *bedrocktypes.ReasoningContentBlockDeltaMemberSignature:
						if !yield(ReasoningDeltaPart{
							ID: contentBlockID,
							ProviderMetadata: ProviderMetadata{
								Bedrock: &BedrockProviderMetadata{
									Signature: reasoningDelta.Value,
								},
							},
						}, nil) {
							return
						}

					case *bedrocktypes.ReasoningContentBlockDeltaMemberRedactedContent:
						if !yield(ReasoningDeltaPart{
							ID: contentBlockID,
							ProviderMetadata: ProviderMetadata{
								Bedrock: &BedrockProviderMetadata{
									RedactedData: base64.StdEncoding.EncodeToString(reasoningDelta.Value),
								},
							},
						}, nil) {
							return
						}
					}

				case *bedrocktypes.ContentBlockDeltaMemberToolUse:
					state := currentToolCalls[contentBlockID]
					if state == nil {
						state = &toolCallState{
							ID: fmt.Sprintf("tool_%s", contentBlockID),
						}
						currentToolCalls[contentBlockID] = state
						currentContentBlockType[contentBlockID] = "tool_use"

						if !yield(ToolInputStartPart{
							ToolCallID: state.ID,
							ToolName:   state.Name,
						}, nil) {
							return
						}
					}

					if delta.Value.Input != nil {
						inputDelta := *delta.Value.Input
						state.Args += inputDelta

						if !yield(ToolInputDeltaPart{
							ToolCallID:     state.ID,
							InputTextDelta: inputDelta,
						}, nil) {
							return
						}
					}
				}

			case *bedrocktypes.ConverseStreamOutputMemberContentBlockStop:
				if !startMessage() {
					return
				}

				contentBlockID := bedrockContentBlockID(event.Value.ContentBlockIndex)

				switch currentContentBlockType[contentBlockID] {
				case "reasoning":
					if !yield(ReasoningEndPart{ID: contentBlockID}, nil) {
						return
					}
				case "text":
					if !yield(TextEndPart{ID: contentBlockID}, nil) {
						return
					}
				case "tool_use":
					state := currentToolCalls[contentBlockID]
					if state != nil {
						var input map[string]any
						if state.Args != "" {
							if err := json.Unmarshal([]byte(state.Args), &input); err != nil {
								yield(nil, fmt.Errorf("unmarshalling tool input for call %s %q: %w", state.ID, state.Args, err))
								return
							}
						}

						if !yield(ToolInputAvailablePart{
							ToolCallID: state.ID,
							ToolName:   state.Name,
							Input:      input,
						}, nil) {
							return
						}
					}
				}

				delete(currentContentBlockType, contentBlockID)
				delete(currentToolCalls, contentBlockID)

			case *bedrocktypes.ConverseStreamOutputMemberMessageStop:
				if !startMessage() {
					return
				}

				finalReason = mapBedrockStopReason(event.Value.StopReason)

			case *bedrocktypes.ConverseStreamOutputMemberMetadata:
				usage = event.Value.Usage
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("bedrock converse stream error: %w", err))
			return
		}

		if !messageStarted {
			if !startMessage() {
				return
			}
		}

		if !yield(FinishStepPart{}, nil) {
			return
		}

		yield(FinishPart{
			FinishReason: finalReason,
		}, nil)
	}

	return dataStream, getUsage
}

func bedrockContentBlockID(index *int32) string {
	if index == nil {
		return "0"
	}

	return fmt.Sprintf("%d", *index)
}

func mapBedrockStopReason(stopReason bedrocktypes.StopReason) FinishReason {
	switch stopReason {
	case "":
		return FinishReasonStop
	case bedrocktypes.StopReasonEndTurn, bedrocktypes.StopReasonStopSequence:
		return FinishReasonStop
	case bedrocktypes.StopReasonToolUse:
		return FinishReasonToolCalls
	case bedrocktypes.StopReasonMaxTokens, bedrocktypes.StopReasonModelContextWindowExceeded:
		return FinishReasonLength
	case bedrocktypes.StopReasonGuardrailIntervened, bedrocktypes.StopReasonContentFiltered:
		return FinishReasonContentFilter
	default:
		return FinishReasonOther
	}
}

// ToolsToBedrock converts internal tool definitions into Bedrock Converse tool specs.
func ToolsToBedrock(tools []Tool) ([]bedrocktypes.Tool, error) {
	bedrockTools := make([]bedrocktypes.Tool, 0, len(tools))

	for _, tool := range tools {
		properties := tool.Schema.Properties
		if properties == nil {
			properties = make(map[string]any)
		}

		schema := map[string]any{
			"type":       "object",
			"properties": properties,
		}
		if len(tool.Schema.Required) > 0 {
			schema["required"] = tool.Schema.Required
		}

		spec := bedrocktypes.ToolSpecification{
			Name:        strPtr(tool.Name),
			Description: strPtrOrNil(tool.Description),
			InputSchema: &bedrocktypes.ToolInputSchemaMemberJson{
				Value: bedrockdocument.NewLazyDocument(schema),
			},
		}

		bedrockTools = append(bedrockTools, &bedrocktypes.ToolMemberToolSpec{
			Value: spec,
		})
	}

	return bedrockTools, nil
}

// MessagesToBedrock converts internal messages into Bedrock Converse messages plus system blocks.
func MessagesToBedrock(messages []Message) ([]bedrocktypes.Message, []bedrocktypes.SystemContentBlock, error) {
	bedrockMessages := make([]bedrocktypes.Message, 0, len(messages))
	systemBlocks := make([]bedrocktypes.SystemContentBlock, 0)

	for _, message := range messages {
		switch message.Role {
		case "system":
			extracted := false
			for _, part := range message.Parts {
				if part.Type != PartTypeText {
					continue
				}
				if strings.TrimSpace(part.Text) == "" {
					continue
				}
				systemBlocks = append(systemBlocks, &bedrocktypes.SystemContentBlockMemberText{
					Value: part.Text,
				})
				extracted = true
			}

			if !extracted && strings.TrimSpace(message.Content) != "" {
				systemBlocks = append(systemBlocks, &bedrocktypes.SystemContentBlockMemberText{
					Value: message.Content,
				})
			}

		case "user":
			content := make([]bedrocktypes.ContentBlock, 0, len(message.Parts)+len(message.Attachments))

			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content = append(content, &bedrocktypes.ContentBlockMemberText{
						Value: part.Text,
					})

				case PartTypeFile:
					block, err := bedrockContentBlockFromBytes(part.MimeType, part.Data, "document")
					if err != nil {
						return nil, nil, err
					}
					content = append(content, block)

				case PartTypeToolInvocation:
					return nil, nil, fmt.Errorf("user message part has type tool-invocation (ID: %s)", message.ID)
				}
			}

			for _, attachment := range message.Attachments {
				data, err := parseDataURLBase64(attachment.URL)
				if err != nil {
					return nil, nil, fmt.Errorf("invalid attachment URL: %w", err)
				}

				block, err := bedrockContentBlockFromBytes(attachment.ContentType, data, attachment.Name)
				if err != nil {
					return nil, nil, err
				}
				content = append(content, block)
			}

			if len(content) > 0 {
				bedrockMessages = append(bedrockMessages, bedrocktypes.Message{
					Role:    bedrocktypes.ConversationRoleUser,
					Content: content,
				})
			}

		case "assistant":
			assistantContent := make([]bedrocktypes.ContentBlock, 0, len(message.Parts))
			flushAssistant := func() {
				if len(assistantContent) == 0 {
					return
				}

				bedrockMessages = append(bedrockMessages, bedrocktypes.Message{
					Role:    bedrocktypes.ConversationRoleAssistant,
					Content: assistantContent,
				})
				assistantContent = nil
			}

			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					assistantContent = append(assistantContent, &bedrocktypes.ContentBlockMemberText{
						Value: part.Text,
					})

				case PartTypeReasoning:
					reasoningText := part.Reasoning
					if reasoningText == "" {
						reasoningText = part.Text
					}

					if part.ProviderMetadata == nil || part.ProviderMetadata.Bedrock == nil {
						continue
					}

					bedrockMetadata := part.ProviderMetadata.Bedrock
					switch {
					case bedrockMetadata.Signature != "":
						signature := bedrockMetadata.Signature
						assistantContent = append(assistantContent, &bedrocktypes.ContentBlockMemberReasoningContent{
							Value: &bedrocktypes.ReasoningContentBlockMemberReasoningText{
								Value: bedrocktypes.ReasoningTextBlock{
									Text:      strPtr(reasoningText),
									Signature: &signature,
								},
							},
						})

					case bedrockMetadata.RedactedData != "":
						decoded, err := base64.StdEncoding.DecodeString(bedrockMetadata.RedactedData)
						if err != nil {
							return nil, nil, fmt.Errorf("invalid bedrock redactedData base64 for message %q: %w", message.ID, err)
						}

						assistantContent = append(assistantContent, &bedrocktypes.ContentBlockMemberReasoningContent{
							Value: &bedrocktypes.ReasoningContentBlockMemberRedactedContent{
								Value: decoded,
							},
						})
					}

				case PartTypeToolInvocation:
					input := part.Input
					if input == nil {
						input = map[string]any{}
					}

					assistantContent = append(assistantContent, &bedrocktypes.ContentBlockMemberToolUse{
						Value: bedrocktypes.ToolUseBlock{
							ToolUseId: strPtr(part.ToolCallID),
							Name:      strPtr(part.ToolName),
							Input:     bedrockdocument.NewLazyDocument(input),
						},
					})

					if part.State != ToolInvocationStateOutputAvailable && part.State != ToolInvocationStateOutputError {
						continue
					}

					flushAssistant()

					resultParts := make([]Part, 0)
					status := bedrocktypes.ToolResultStatusSuccess

					if part.State == ToolInvocationStateOutputError {
						status = bedrocktypes.ToolResultStatusError
						resultParts = []Part{{Type: PartTypeText, Text: part.ErrorText}}
					} else {
						var err error
						resultParts, err = toolResultToParts(part.Output)
						if err != nil {
							return nil, nil, fmt.Errorf("failed to convert tool call result to parts: %w", err)
						}
					}

					toolResultContent := make([]bedrocktypes.ToolResultContentBlock, 0, len(resultParts))
					for _, resultPart := range resultParts {
						switch resultPart.Type {
						case PartTypeText:
							toolResultContent = append(toolResultContent, &bedrocktypes.ToolResultContentBlockMemberText{
								Value: resultPart.Text,
							})

						case PartTypeFile:
							resultBlock, err := bedrockToolResultContentBlockFromBytes(resultPart.MimeType, resultPart.Data, "document")
							if err != nil {
								return nil, nil, err
							}
							toolResultContent = append(toolResultContent, resultBlock)
						}
					}

					bedrockMessages = append(bedrockMessages, bedrocktypes.Message{
						Role: bedrocktypes.ConversationRoleUser,
						Content: []bedrocktypes.ContentBlock{
							&bedrocktypes.ContentBlockMemberToolResult{
								Value: bedrocktypes.ToolResultBlock{
									ToolUseId: strPtr(part.ToolCallID),
									Content:   toolResultContent,
									Status:    status,
								},
							},
						},
					})
				}
			}

			flushAssistant()

		default:
			return nil, nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}
	}

	return bedrockMessages, systemBlocks, nil
}

func parseDataURLBase64(dataURL string) ([]byte, error) {
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("expected data URL with base64 payload")
	}

	decoded, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, fmt.Errorf("decode base64 payload: %w", err)
	}

	return decoded, nil
}

func bedrockContentBlockFromBytes(mimeType string, data []byte, name string) (bedrocktypes.ContentBlock, error) {
	if imageFormat, ok := bedrockImageFormatFromMime(mimeType); ok {
		return &bedrocktypes.ContentBlockMemberImage{
			Value: bedrocktypes.ImageBlock{
				Format: imageFormat,
				Source: &bedrocktypes.ImageSourceMemberBytes{Value: data},
			},
		}, nil
	}

	documentFormat, ok := bedrockDocumentFormatFromMime(mimeType)
	if !ok {
		return nil, fmt.Errorf("unsupported bedrock file mime type: %s", mimeType)
	}

	documentName := name
	if strings.TrimSpace(documentName) == "" {
		documentName = "document"
	}

	return &bedrocktypes.ContentBlockMemberDocument{
		Value: bedrocktypes.DocumentBlock{
			Name:   strPtr(documentName),
			Format: documentFormat,
			Source: &bedrocktypes.DocumentSourceMemberBytes{Value: data},
		},
	}, nil
}

func bedrockToolResultContentBlockFromBytes(mimeType string, data []byte, name string) (bedrocktypes.ToolResultContentBlock, error) {
	if imageFormat, ok := bedrockImageFormatFromMime(mimeType); ok {
		return &bedrocktypes.ToolResultContentBlockMemberImage{
			Value: bedrocktypes.ImageBlock{
				Format: imageFormat,
				Source: &bedrocktypes.ImageSourceMemberBytes{Value: data},
			},
		}, nil
	}

	documentFormat, ok := bedrockDocumentFormatFromMime(mimeType)
	if !ok {
		return nil, fmt.Errorf("unsupported bedrock tool-result mime type: %s", mimeType)
	}

	documentName := name
	if strings.TrimSpace(documentName) == "" {
		documentName = "document"
	}

	return &bedrocktypes.ToolResultContentBlockMemberDocument{
		Value: bedrocktypes.DocumentBlock{
			Name:   strPtr(documentName),
			Format: documentFormat,
			Source: &bedrocktypes.DocumentSourceMemberBytes{Value: data},
		},
	}, nil
}

func bedrockImageFormatFromMime(mimeType string) (bedrocktypes.ImageFormat, bool) {
	switch strings.ToLower(strings.TrimSpace(mimeType)) {
	case "image/png":
		return bedrocktypes.ImageFormatPng, true
	case "image/jpeg", "image/jpg":
		return bedrocktypes.ImageFormatJpeg, true
	case "image/gif":
		return bedrocktypes.ImageFormatGif, true
	case "image/webp":
		return bedrocktypes.ImageFormatWebp, true
	default:
		return "", false
	}
}

func bedrockDocumentFormatFromMime(mimeType string) (bedrocktypes.DocumentFormat, bool) {
	switch strings.ToLower(strings.TrimSpace(mimeType)) {
	case "application/pdf":
		return bedrocktypes.DocumentFormatPdf, true
	case "text/csv":
		return bedrocktypes.DocumentFormatCsv, true
	case "application/msword":
		return bedrocktypes.DocumentFormatDoc, true
	case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
		return bedrocktypes.DocumentFormatDocx, true
	case "application/vnd.ms-excel":
		return bedrocktypes.DocumentFormatXls, true
	case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
		return bedrocktypes.DocumentFormatXlsx, true
	case "text/html":
		return bedrocktypes.DocumentFormatHtml, true
	case "text/plain":
		return bedrocktypes.DocumentFormatTxt, true
	case "text/markdown", "text/x-markdown", "application/markdown":
		return bedrocktypes.DocumentFormatMd, true
	default:
		return "", false
	}
}

func strPtr(value string) *string {
	return &value
}

func strPtrOrNil(value string) *string {
	if strings.TrimSpace(value) == "" {
		return nil
	}

	return &value
}
