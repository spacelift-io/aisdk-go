package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/packages/ssestream"
)

// ToolsToOpenAI converts the tool format to OpenAI's API format.
func ToolsToOpenAI(tools []Tool) []openai.ChatCompletionToolParam {
	openaiTools := []openai.ChatCompletionToolParam{}
	for _, tool := range tools {
		var schemaParams map[string]any
		if tool.Schema.Properties != nil {
			schemaParams = map[string]any{
				"type":       "object",
				"properties": tool.Schema.Properties,
			}
			if len(tool.Schema.Required) > 0 {
				schemaParams["required"] = tool.Schema.Required
			}
		}
		openaiTools = append(openaiTools, openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: param.NewOpt(tool.Description),
				Parameters:  schemaParams,
			},
		})
	}
	return openaiTools
}

// MessagesToOpenAI converts internal message format to OpenAI's API format.
func MessagesToOpenAI(messages []Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	openaiMessages := []openai.ChatCompletionMessageParamUnion{}

	for _, message := range messages {
		switch message.Role {
		case "system":
			openaiMessages = append(openaiMessages, openai.SystemMessage(message.Content))
		case "user":
			content := []openai.ChatCompletionContentPartUnionParam{}
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content = append(content, openai.ChatCompletionContentPartUnionParam{
						OfText: &openai.ChatCompletionContentPartTextParam{
							Text: part.Text,
						},
					})
				case PartTypeFile:
					content = append(content, openai.ChatCompletionContentPartUnionParam{
						OfImageURL: &openai.ChatCompletionContentPartImageParam{
							ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
								URL: fmt.Sprintf("data:%s;base64,%s", part.MimeType, base64.StdEncoding.EncodeToString(part.Data)),
							},
						},
					})
				}
			}

			for _, attachment := range message.Attachments {
				content = append(content, openai.ChatCompletionContentPartUnionParam{
					OfImageURL: &openai.ChatCompletionContentPartImageParam{
						ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
							URL: attachment.URL,
						},
					},
				})
			}

			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfArrayOfContentParts: content,
					},
				},
			})
		case "assistant":
			content := &openai.ChatCompletionAssistantMessageParam{}

			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content.Content.OfArrayOfContentParts = append(content.Content.OfArrayOfContentParts, openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
						OfText: &openai.ChatCompletionContentPartTextParam{
							Text: part.Text,
						},
					})
				case PartTypeToolInvocation:
					if part.Input == nil {
						part.Input = make(map[string]any)
					}
					argsJSON, err := json.Marshal(part.Input)
					if err != nil {
						return nil, fmt.Errorf("marshalling tool input for call %s: %w", part.ToolCallID, err)
					}
					content.ToolCalls = append(content.ToolCalls, openai.ChatCompletionMessageToolCallParam{
						ID: part.ToolCallID,
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      part.ToolName,
							Arguments: string(argsJSON),
						},
					})

					if part.State != ToolInvocationStateOutputAvailable && part.State != ToolInvocationStateOutputError {
						continue
					}

					openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
						OfAssistant: content,
					})
					content = &openai.ChatCompletionAssistantMessageParam{}

					parts := []openai.ChatCompletionContentPartTextParam{}

					var resultParts []Part
					if part.State == ToolInvocationStateOutputError {
						resultParts = []Part{{Type: PartTypeText, Text: part.ErrorText}}
					} else {
						var err error
						resultParts, err = toolResultToParts(part.Output)
						if err != nil {
							return nil, fmt.Errorf("failed to convert tool call result to parts: %w", err)
						}
					}
					for _, resultPart := range resultParts {
						switch resultPart.Type {
						case PartTypeText:
							parts = append(parts, openai.ChatCompletionContentPartTextParam{
								Text: resultPart.Text,
							})
						case PartTypeFile:
							// Unfortunately, OpenAI doesn't support file content in tool messages.
							parts = append(parts, openai.ChatCompletionContentPartTextParam{
								Text: "File content was provided as a tool result, but is not supported by OpenAI.",
							})
							continue
						}
					}

					openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
						OfTool: &openai.ChatCompletionToolMessageParam{
							ToolCallID: part.ToolCallID,
							Content: openai.ChatCompletionToolMessageParamContentUnion{
								OfArrayOfContentParts: parts,
							},
						},
					})
				}
			}

			if len(content.Content.OfArrayOfContentParts) > 0 || len(content.ToolCalls) > 0 {
				openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
					OfAssistant: content,
				})
			}
		}
	}

	return openaiMessages, nil
}

// OpenAIToDataStream pipes an OpenAI stream to a DataStream.
func OpenAIToDataStream(stream *ssestream.Stream[openai.ChatCompletionChunk]) (DataStream, func() openai.CompletionUsage) {
	usage := openai.CompletionUsage{}
	getUsage := func() openai.CompletionUsage {
		return usage
	}

	dataStream := func(yield func(DataStreamPart, error) bool) {
		var lastFinishReason string
		var messageStarted bool
		var currentContentBlockID int
		currentContentBlockIDText := "0"
		currentContentBlockType := ""
		bumpContentBlockID := func() {
			currentContentBlockID++
			currentContentBlockIDText = fmt.Sprintf("%d", currentContentBlockID)
		}

		type toolCallState struct {
			ID      string
			Name    string
			Args    string
			Started bool
		}

		toolCalls := map[int64]*toolCallState{}
		var sawToolCall bool

		startMessage := func(chunk *openai.ChatCompletionChunk) bool {
			if messageStarted {
				return true
			}
			messageStarted = true
			messageID := ""
			if chunk != nil {
				messageID = chunk.ID
			}
			if !yield(MessageStartPart{MessageID: messageID}, nil) {
				return false
			}
			if !yield(StartStepStreamPart{}, nil) {
				return false
			}
			return true
		}

		closeTextBlock := func() bool {
			if currentContentBlockType != "text" {
				return true
			}
			if !yield(TextEndPart{ID: currentContentBlockIDText}, nil) {
				return false
			}
			currentContentBlockType = ""
			bumpContentBlockID()
			return true
		}

		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		for stream.Next() {
			chunk := stream.Current()
			if chunk.JSON.Usage.Valid() {
				usage = chunk.Usage
			}

			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			if choice.FinishReason != "" {
				lastFinishReason = choice.FinishReason
			}

			if !startMessage(&chunk) {
				return
			}

			textDelta := choice.Delta.Content
			if choice.Delta.Refusal != "" {
				textDelta += choice.Delta.Refusal
			}

			if textDelta != "" {
				if currentContentBlockType != "text" {
					if !yield(TextStartPart{ID: currentContentBlockIDText}, nil) {
						return
					}
					currentContentBlockType = "text"
				}
				if !yield(TextDeltaPart{
					ID:    currentContentBlockIDText,
					Delta: textDelta,
				}, nil) {
					return
				}
			}

			if len(choice.Delta.ToolCalls) > 0 {
				sawToolCall = true
				if !closeTextBlock() {
					return
				}
			}

			for _, toolCallDelta := range choice.Delta.ToolCalls {
				state, ok := toolCalls[toolCallDelta.Index]
				if !ok {
					state = &toolCallState{}
					toolCalls[toolCallDelta.Index] = state
				}
				if toolCallDelta.ID != "" {
					state.ID = toolCallDelta.ID
				}
				if toolCallDelta.Function.Name != "" {
					state.Name = toolCallDelta.Function.Name
				}
				if toolCallDelta.Function.Arguments != "" {
					state.Args += toolCallDelta.Function.Arguments
				}

				if !state.Started {
					if state.ID == "" {
						state.ID = fmt.Sprintf("call_%d", toolCallDelta.Index)
					}
					if !yield(ToolInputStartPart{
						ToolCallID: state.ID,
						ToolName:   state.Name,
					}, nil) {
						return
					}
					state.Started = true
				}

				if toolCallDelta.Function.Arguments != "" {
					if !yield(ToolInputDeltaPart{
						ToolCallID:     state.ID,
						InputTextDelta: toolCallDelta.Function.Arguments,
					}, nil) {
						return
					}
				}
			}

		}

		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		if currentContentBlockType == "text" {
			if !yield(TextEndPart{ID: currentContentBlockIDText}, nil) {
				return
			}
		}

		if len(toolCalls) > 0 {
			indices := make([]int, 0, len(toolCalls))
			for idx := range toolCalls {
				indices = append(indices, int(idx))
			}
			sort.Ints(indices)
			for _, idx := range indices {
				state := toolCalls[int64(idx)]
				if state == nil {
					continue
				}
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

		var finishReason FinishReason

		if lastFinishReason != "" {
			switch lastFinishReason {
			case "stop":
				finishReason = FinishReasonStop
			case "length":
				finishReason = FinishReasonLength
			case "content_filter":
				finishReason = FinishReasonContentFilter
			case "tool_calls", "function_call":
				finishReason = FinishReasonToolCalls
			default:
				finishReason = FinishReasonOther
			}
		}

		if finishReason == "" && sawToolCall {
			finishReason = FinishReasonToolCalls
		}
		if finishReason == "" {
			finishReason = FinishReasonStop
		}

		if !messageStarted {
			if !yield(MessageStartPart{}, nil) {
				return
			}
			if !yield(StartStepStreamPart{}, nil) {
				return
			}
		}

		if !yield(FinishStepPart{}, nil) {
			return
		}

		yield(FinishPart{
			FinishReason: finishReason,
		}, nil)
	}

	return dataStream, getUsage
}
