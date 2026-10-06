package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/openai/openai-go/responses"
)

// ToolsToOpenAIResponses converts the tool format to Responses API function tools.
func ToolsToOpenAIResponses(tools []Tool) []responses.ToolUnionParam {
	responseTools := []responses.ToolUnionParam{}
	for _, tool := range tools {
		// Parameters is a required field, so a tool without arguments still gets
		// an empty object schema.
		properties := tool.Schema.Properties
		if properties == nil {
			properties = map[string]any{}
		}
		schemaParams := map[string]any{
			"type":       "object",
			"properties": properties,
		}
		if len(tool.Schema.Required) > 0 {
			schemaParams["required"] = tool.Schema.Required
		}
		responseTools = append(responseTools, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.Name,
				Description: param.NewOpt(tool.Description),
				Parameters:  schemaParams,
				// Mirrors the Vercel SDK: strict mode is the server default and
				// rejects schemas with optional properties, which most tools have.
				Strict: param.NewOpt(false),
			},
		})
	}
	return responseTools
}

// MessagesToOpenAIResponses converts internal messages into Responses API input
// items. Reasoning items are replayed with their encrypted content so the model
// keeps its reasoning across turns without OpenAI storing the conversation.
func MessagesToOpenAIResponses(messages []Message) (responses.ResponseInputParam, error) {
	openaiInput := responses.ResponseInputParam{}

	for _, message := range messages {
		switch message.Role {
		case "system":
			openaiInput = append(openaiInput, responses.ResponseInputItemParamOfMessage(message.Content, responses.EasyInputMessageRoleSystem))
		case "user":
			content := responses.ResponseInputMessageContentListParam{}
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content = append(content, responses.ResponseInputContentUnionParam{
						OfInputText: &responses.ResponseInputTextParam{Text: part.Text},
					})
				case PartTypeFile:
					content = append(content, responses.ResponseInputContentUnionParam{
						OfInputImage: &responses.ResponseInputImageParam{
							ImageURL: param.NewOpt(fmt.Sprintf("data:%s;base64,%s", part.MimeType, base64.StdEncoding.EncodeToString(part.Data))),
							Detail:   responses.ResponseInputImageDetailAuto,
						},
					})
				}
			}
			for _, attachment := range message.Attachments {
				content = append(content, responses.ResponseInputContentUnionParam{
					OfInputImage: &responses.ResponseInputImageParam{
						ImageURL: param.NewOpt(attachment.URL),
						Detail:   responses.ResponseInputImageDetailAuto,
					},
				})
			}
			openaiInput = append(openaiInput, responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRoleUser))
		case "assistant":
			items, err := assistantMessageToOpenAIResponses(message)
			if err != nil {
				return nil, err
			}
			openaiInput = append(openaiInput, items...)
		}
	}

	return openaiInput, nil
}

func assistantMessageToOpenAIResponses(message Message) (responses.ResponseInputParam, error) {
	items := responses.ResponseInputParam{}
	// Summary parts of one reasoning item arrive as separate parts that share an
	// item ID and go back as one item.
	reasoningItems := map[string]int{}
	var toolOutputs responses.ResponseInputParam

	// One assistant message holds every step of a tool loop. A tool output goes
	// right after its step, before the reasoning and text of the next one.
	flushToolOutputs := func() {
		items = append(items, toolOutputs...)
		toolOutputs = nil
	}

	for _, part := range message.Parts {
		switch part.Type {
		case PartTypeStepStart:
			flushToolOutputs()
		case PartTypeText:
			flushToolOutputs()
			items = append(items, responses.ResponseInputItemParamOfMessage(part.Text, responses.EasyInputMessageRoleAssistant))
		case PartTypeReasoning:
			// Mirrors the Vercel SDK: a reasoning part is replayed only with the
			// metadata the Responses API issued for it, and the item ID is a
			// required field of a reasoning input item.
			if part.ProviderMetadata == nil || part.ProviderMetadata.OpenAI == nil || part.ProviderMetadata.OpenAI.ItemID == "" {
				continue
			}

			openaiMetadata := part.ProviderMetadata.OpenAI
			var summary []responses.ResponseReasoningItemSummaryParam
			if text := reasoningText(part); text != "" {
				summary = append(summary, responses.ResponseReasoningItemSummaryParam{Text: text})
			}

			index, ok := reasoningItems[openaiMetadata.ItemID]
			if !ok {
				flushToolOutputs()
				index = len(items)
				reasoningItems[openaiMetadata.ItemID] = index
				// Summary is a required field, even when the model wrote none.
				items = append(items, responses.ResponseInputItemParamOfReasoning(openaiMetadata.ItemID, []responses.ResponseReasoningItemSummaryParam{}))
			}
			item := items[index].OfReasoning
			item.Summary = append(item.Summary, summary...)
			if openaiMetadata.ReasoningEncryptedContent != "" {
				item.EncryptedContent = param.NewOpt(openaiMetadata.ReasoningEncryptedContent)
			}
		case PartTypeToolInvocation:
			// Mirrors the Vercel SDK: a call whose input never finished streaming
			// has no output to pair with, and an unpaired call fails the request.
			if part.State == ToolStateInputStreaming {
				continue
			}
			if part.Input == nil {
				part.Input = make(map[string]any)
			}
			argsJSON, err := json.Marshal(part.Input)
			if err != nil {
				return nil, fmt.Errorf("marshalling tool input for call %s: %w", part.ToolCallID, err)
			}
			items = append(items, responses.ResponseInputItemParamOfFunctionCall(string(argsJSON), part.ToolCallID, part.ToolName))

			denied := isDeniedToolPart(part)
			if part.State != ToolStateOutputAvailable && part.State != ToolStateOutputError && !denied {
				continue
			}

			var resultParts []Part
			if denied {
				resultParts = []Part{{Type: PartTypeText, Text: deniedToolResultReason(part)}}
			} else if part.State == ToolStateOutputError {
				resultParts = []Part{{Type: PartTypeText, Text: part.ErrorText}}
			} else {
				var err error
				resultParts, err = toolResultToParts(part.Output)
				if err != nil {
					return nil, fmt.Errorf("failed to convert tool call result to parts: %w", err)
				}
			}

			texts := make([]string, 0, len(resultParts))
			for _, resultPart := range resultParts {
				switch resultPart.Type {
				case PartTypeText:
					texts = append(texts, resultPart.Text)
				case PartTypeFile:
					// A function call output is a string, so file content has no place in it.
					texts = append(texts, "File content was provided as a tool result, but is not supported by OpenAI.")
				}
			}
			toolOutputs = append(toolOutputs, responses.ResponseInputItemParamOfFunctionCallOutput(part.ToolCallID, strings.Join(texts, "\n")))
		}
	}

	// The encrypted content arrives with the last summary part of an item. An
	// item without it comes from a stream cut off mid-reasoning, and with nothing
	// stored on OpenAI's side its ID alone fails the request.
	complete := make(responses.ResponseInputParam, 0, len(items)+len(toolOutputs))
	for _, item := range items {
		if item.OfReasoning != nil && !item.OfReasoning.EncryptedContent.Valid() {
			continue
		}
		complete = append(complete, item)
	}

	return append(complete, toolOutputs...), nil
}

// OpenAIResponsesToDataStream pipes a Responses API stream to a DataStream.
func OpenAIResponsesToDataStream(stream *ssestream.Stream[responses.ResponseStreamEventUnion]) (DataStream, func() responses.ResponseUsage) {
	usage := responses.ResponseUsage{}
	getUsage := func() responses.ResponseUsage {
		return usage
	}

	dataStream := func(yield func(DataStreamPart, error) bool) {
		type toolCallState struct {
			ID   string
			Name string
		}

		var messageStarted bool
		var sawToolCall bool
		var finished bool
		var incompleteReason string
		toolCalls := map[int64]*toolCallState{}
		// The summary index of the reasoning part currently open, per reasoning item.
		reasoningSummaryIndex := map[string]int64{}

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

		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		for stream.Next() {
			if !startMessage() {
				return
			}

			switch event := stream.Current().AsAny().(type) {
			case responses.ResponseOutputItemAddedEvent:
				switch item := event.Item.AsAny().(type) {
				case responses.ResponseOutputMessage:
					if !yield(TextStartPart{ID: item.ID}, nil) {
						return
					}
				case responses.ResponseReasoningItem:
					reasoningSummaryIndex[item.ID] = 0
					if !yield(ReasoningStartPart{ID: openaiReasoningPartID(item.ID, 0)}, nil) {
						return
					}
				case responses.ResponseFunctionToolCall:
					toolCalls[event.OutputIndex] = &toolCallState{ID: item.CallID, Name: item.Name}
					if !yield(ToolInputStartPart{ToolCallID: item.CallID, ToolName: item.Name}, nil) {
						return
					}
				}

			case responses.ResponseTextDeltaEvent:
				if !yield(TextDeltaPart{ID: event.ItemID, Delta: event.Delta}, nil) {
					return
				}

			case responses.ResponseRefusalDeltaEvent:
				if !yield(TextDeltaPart{ID: event.ItemID, Delta: event.Delta}, nil) {
					return
				}

			case responses.ResponseReasoningSummaryPartAddedEvent:
				current, ok := reasoningSummaryIndex[event.ItemID]
				if !ok || event.SummaryIndex == 0 {
					continue
				}
				if !yield(ReasoningEndPart{ID: openaiReasoningPartID(event.ItemID, current)}, nil) {
					return
				}
				reasoningSummaryIndex[event.ItemID] = event.SummaryIndex
				if !yield(ReasoningStartPart{ID: openaiReasoningPartID(event.ItemID, event.SummaryIndex)}, nil) {
					return
				}

			case responses.ResponseReasoningSummaryTextDeltaEvent:
				if !yield(ReasoningDeltaPart{
					ID:               openaiReasoningPartID(event.ItemID, event.SummaryIndex),
					Delta:            event.Delta,
					ProviderMetadata: ProviderMetadata{OpenAI: &OpenAIProviderMetadata{ItemID: event.ItemID}},
				}, nil) {
					return
				}

			case responses.ResponseFunctionCallArgumentsDeltaEvent:
				state, ok := toolCalls[event.OutputIndex]
				if !ok {
					continue
				}
				if !yield(ToolInputDeltaPart{ToolCallID: state.ID, InputTextDelta: event.Delta}, nil) {
					return
				}

			case responses.ResponseOutputItemDoneEvent:
				switch item := event.Item.AsAny().(type) {
				case responses.ResponseOutputMessage:
					if !yield(TextEndPart{ID: item.ID}, nil) {
						return
					}
				case responses.ResponseReasoningItem:
					current, ok := reasoningSummaryIndex[item.ID]
					if !ok {
						continue
					}
					delete(reasoningSummaryIndex, item.ID)
					// The encrypted content arrives only with the finished item, so the
					// last summary part carries it for the replay on the next turn.
					if !yield(ReasoningDeltaPart{
						ID: openaiReasoningPartID(item.ID, current),
						ProviderMetadata: ProviderMetadata{OpenAI: &OpenAIProviderMetadata{
							ItemID:                    item.ID,
							ReasoningEncryptedContent: item.EncryptedContent,
						}},
					}, nil) {
						return
					}
					if !yield(ReasoningEndPart{ID: openaiReasoningPartID(item.ID, current)}, nil) {
						return
					}
				case responses.ResponseFunctionToolCall:
					delete(toolCalls, event.OutputIndex)
					sawToolCall = true

					var input map[string]any
					var inputErr error
					if item.Arguments != "" {
						inputErr = json.Unmarshal([]byte(item.Arguments), &input)
					}
					if inputErr != nil {
						// Mirrors the Vercel SDK: malformed tool input never errors the
						// stream. The error parts go out and the turn finishes normally.
						errorText := fmt.Sprintf("unmarshalling tool input for call %s: %s", item.CallID, inputErr)
						if !yield(ToolInputErrorPart{
							ToolCallID: item.CallID,
							ToolName:   item.Name,
							Input:      item.Arguments,
							ErrorText:  errorText,
						}, nil) {
							return
						}
						if !yield(ToolOutputErrorPart{ToolCallID: item.CallID, ErrorText: errorText}, nil) {
							return
						}
						continue
					}
					if !yield(ToolInputAvailablePart{
						ToolCallID: item.CallID,
						ToolName:   item.Name,
						Input:      input,
					}, nil) {
						return
					}
				}

			case responses.ResponseCompletedEvent:
				finished = true
				usage = event.Response.Usage
				incompleteReason = event.Response.IncompleteDetails.Reason

			case responses.ResponseIncompleteEvent:
				finished = true
				usage = event.Response.Usage
				incompleteReason = event.Response.IncompleteDetails.Reason

			case responses.ResponseFailedEvent:
				usage = event.Response.Usage
				message := event.Response.Error.Message
				if message == "" {
					message = "response failed"
				}
				yield(nil, fmt.Errorf("openai response %s failed: %s", event.Response.ID, message))
				return

			case responses.ResponseErrorEvent:
				yield(nil, fmt.Errorf("openai stream error %s: %s", event.Code, event.Message))
				return
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		if !startMessage() {
			return
		}

		if !yield(FinishStepPart{}, nil) {
			return
		}

		// Mirrors the Vercel SDK: a stream that closes before the response
		// finished is not a normal stop.
		finishReason := FinishReasonOther
		if finished {
			finishReason = mapOpenAIResponsesFinishReason(incompleteReason, sawToolCall)
		}

		yield(FinishPart{
			FinishReason: finishReason,
		}, nil)
	}

	return dataStream, getUsage
}

// openaiReasoningPartID keys a reasoning part by its item and summary index,
// since one reasoning item streams several summaries.
func openaiReasoningPartID(itemID string, summaryIndex int64) string {
	return fmt.Sprintf("%s:%d", itemID, summaryIndex)
}

// mapOpenAIResponsesFinishReason mirrors the Vercel SDK. A finished response has
// no stop reason of its own, and only an incomplete one comes with the reason.
func mapOpenAIResponsesFinishReason(incompleteReason string, sawToolCall bool) FinishReason {
	switch incompleteReason {
	case "":
		if sawToolCall {
			return FinishReasonToolCalls
		}
		return FinishReasonStop
	case "max_output_tokens":
		return FinishReasonLength
	case "content_filter":
		return FinishReasonContentFilter
	default:
		if sawToolCall {
			return FinishReasonToolCalls
		}
		return FinishReasonOther
	}
}
