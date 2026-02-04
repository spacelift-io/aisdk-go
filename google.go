package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"iter"
	"strings"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

// GoogleStreamIterator defines the interface for iterating over Google AI stream responses.
// This allows for mocking in tests.
type GoogleStreamIterator interface {
	Next() (*genai.GenerateContentResponse, error)
}

func ToolsToGoogle(tools []Tool) ([]*genai.Tool, error) {
	functionDeclarations := []*genai.FunctionDeclaration{}

	var propertyToSchema func(property map[string]any) (*genai.Schema, error)
	propertyToSchema = func(property map[string]any) (*genai.Schema, error) {
		schema := &genai.Schema{
			Properties: make(map[string]*genai.Schema),
		}

		typeRaw, ok := property["type"]
		if ok {
			typ, ok := typeRaw.(string)
			if !ok {
				return nil, fmt.Errorf("type is not a string: %T", typeRaw)
			}
			schema.Type = genai.Type(strings.ToUpper(typ))
		}

		descriptionRaw, ok := property["description"]
		if ok {
			description, ok := descriptionRaw.(string)
			if !ok {
				return nil, fmt.Errorf("description is not a string: %T", descriptionRaw)
			}
			schema.Description = description
		}

		propertiesRaw, ok := property["properties"]
		if ok {
			properties, ok := propertiesRaw.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("properties is not a map[string]any: %T", propertiesRaw)
			}

			for key, value := range properties {
				propMap, ok := value.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("property %q is not a map[string]any: %T", key, value)
				}
				subschema, err := propertyToSchema(propMap)
				if err != nil {
					return nil, fmt.Errorf("property %q has non-object properties: %w", key, err)
				}
				schema.Properties[key] = subschema
			}
		}

		itemsRaw, ok := property["items"]
		if ok {
			items, ok := itemsRaw.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("items is not a map[string]any: %T", itemsRaw)
			}
			subschema, err := propertyToSchema(items)
			if err != nil {
				return nil, fmt.Errorf("items has non-object properties: %w", err)
			}
			schema.Items = subschema
		}

		return schema, nil
	}

	for _, tool := range tools {
		var schema *genai.Schema
		if tool.Schema.Properties != nil {
			schema = &genai.Schema{
				Type:       genai.TypeObject,
				Properties: make(map[string]*genai.Schema),
				Required:   tool.Schema.Required,
			}

			for key, value := range tool.Schema.Properties {
				propMap, ok := value.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("property %q is not a map[string]any: %T", key, value)
				}
				subschema, err := propertyToSchema(propMap)
				if err != nil {
					return nil, fmt.Errorf("property %q has non-object properties: %w", key, err)
				}
				schema.Properties[key] = subschema
			}
		}

		functionDeclarations = append(functionDeclarations, &genai.FunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  schema,
		})
	}
	return []*genai.Tool{{
		FunctionDeclarations: functionDeclarations,
	}}, nil
}

// MessagesToGoogle converts internal message format to Google's genai.Content slice.
// System messages are ignored.
func MessagesToGoogle(messages []Message) ([]*genai.Content, error) {
	googleContents := []*genai.Content{}

	for _, message := range messages {
		switch message.Role {
		case "system":
			// System messages are ignored for Google's main message history.
			// They are handled separately via SystemInstruction.

		case "user":
			content := &genai.Content{
				Role: "user",
			}
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content.Parts = append(content.Parts, &genai.Part{Text: part.Text})
				case PartTypeFile:
					content.Parts = append(content.Parts, &genai.Part{InlineData: &genai.Blob{
						Data:     part.Data,
						MIMEType: part.MimeType,
					}})
				}
			}

			for _, attachment := range message.Attachments {
				parts := strings.SplitN(attachment.URL, ",", 2)
				if len(parts) != 2 {
					return nil, fmt.Errorf("invalid attachment URL: %s", attachment.URL)
				}
				decoded, err := base64.StdEncoding.DecodeString(parts[1])
				if err != nil {
					return nil, fmt.Errorf("failed to decode attachment: %w", err)
				}
				content.Parts = append(content.Parts, &genai.Part{InlineData: &genai.Blob{
					Data:     decoded,
					MIMEType: attachment.ContentType,
				}})
			}

			googleContents = append(googleContents, content)
		case "assistant":
			content := &genai.Content{
				Role: "model",
			}
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					content.Parts = append(content.Parts, &genai.Part{
						Text: part.Text,
					})
				case PartTypeToolInvocation:
					argsMap, ok := part.Input.(map[string]any)
					if !ok && part.Input != nil { // Allow nil args
						return nil, fmt.Errorf("tool call args for %s are not map[string]any: %T", part.ToolName, part.Input)
					}
					fc := genai.FunctionCall{
						ID:   part.ToolCallID,
						Name: part.ToolName,
						Args: argsMap,
					}
					// Create the part with function call
					genaiPart := &genai.Part{
						FunctionCall: &fc,
					}

					// Include thought signature if present in provider metadata
					if part.ProviderMetadata != nil && part.ProviderMetadata.Google != nil {
						genaiPart.ThoughtSignature = part.ProviderMetadata.Google.ThoughtSignature
					}

					content.Parts = append(content.Parts, genaiPart)

					if part.State != ToolInvocationStateOutputAvailable && part.State != ToolInvocationStateOutputError {
						continue
					}

					googleContents = append(googleContents, content)
					content = &genai.Content{
						Role: "model",
					}

					googleParts := []*genai.Part{}

					var parts []Part
					if part.State == ToolInvocationStateOutputError {
						parts = []Part{{Type: PartTypeText, Text: part.ErrorText}}
					} else {
						var err error
						parts, err = toolResultToParts(part.Output)
						if err != nil {
							return nil, fmt.Errorf("failed to convert tool call result to parts: %w", err)
						}
					}
					for _, part := range parts {
						switch part.Type {
						case PartTypeText:
							googleParts = append(googleParts, &genai.Part{
								Text: part.Text,
							})
						case PartTypeFile:
							googleParts = append(googleParts, &genai.Part{
								InlineData: &genai.Blob{
									Data:     part.Data,
									MIMEType: part.MimeType,
								}},
							)
						}
					}

					fr := genai.FunctionResponse{
						Name:     part.ToolName,
						ID:       part.ToolCallID,
						Response: map[string]any{"output": googleParts},
					}
					content.Parts = append(content.Parts, &genai.Part{FunctionResponse: &fr})
				}
			}

			googleContents = append(googleContents, content)
		default:
			return nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}
	}

	return googleContents, nil
}

// GoogleToDataStream pipes a Google AI stream to a DataStream.
func GoogleToDataStream(stream iter.Seq2[*genai.GenerateContentResponse, error]) (DataStream, func() *genai.GenerateContentResponseUsageMetadata) {
	var usage *genai.GenerateContentResponseUsageMetadata
	getUsage := func() *genai.GenerateContentResponseUsageMetadata {
		return usage
	}

	dataStream := func(yield func(DataStreamPart, error) bool) {
		finalReason := FinishReasonUnknown
		var lastResp *genai.GenerateContentResponse
		var messageStarted bool
		var currentContentBlockID int
		currentContentBlockIDText := "0"
		currentContentBlockType := ""
		var thoughtSignature []byte // Persists across chunks for parallel function calls
		bumpContentBlockID := func() {
			currentContentBlockID++
			currentContentBlockIDText = fmt.Sprintf("%d", currentContentBlockID)
		}

		for resp, err := range stream {
			if err != nil {
				yield(nil, err)
				return
			}

			if resp == nil {
				continue
			}

			lastResp = resp

			if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
				continue
			}
			cand := resp.Candidates[0]
			content := cand.Content

			if !messageStarted {
				messageStarted = true
				if !yield(MessageStartPart{}, nil) {
					return
				}
				if !yield(StartStepStreamPart{}, nil) {
					return
				}
			}

			for _, part := range content.Parts {
				if part.FunctionCall != nil {
					fc := part.FunctionCall
					// Google's stream often sends the full FunctionCall in one part.
					// We translate this into start and delta parts for the generic stream.
					toolCallID := fc.ID
					if toolCallID == "" {
						toolCallID = uuid.New().String()
					}

					if currentContentBlockType == "text" {
						if !yield(TextEndPart{ID: currentContentBlockIDText}, nil) {
							return
						}
						currentContentBlockType = ""
						bumpContentBlockID()
					}
					if currentContentBlockType == "reasoning" {
						if !yield(ReasoningEndPart{ID: currentContentBlockIDText}, nil) {
							return
						}
						currentContentBlockType = ""
						bumpContentBlockID()
					}

					if !yield(ToolInputStartPart{
						ToolCallID: toolCallID,
						ToolName:   fc.Name,
					}, nil) {
						return
					}

					var argsJSON string
					if fc.Args != nil {
						if jsonBytes, err := json.Marshal(fc.Args); err == nil {
							argsJSON = string(jsonBytes)
						}
					}

					if argsJSON != "" {
						if !yield(ToolInputDeltaPart{
							ToolCallID:     toolCallID,
							InputTextDelta: argsJSON,
						}, nil) {
							return
						}
					}

					// Capture thought signature - Google only sends it on the first parallel function call
					if len(part.ThoughtSignature) > 0 {
						thoughtSignature = part.ThoughtSignature
					}

					var providerMetadata ProviderMetadata
					if len(thoughtSignature) > 0 {
						providerMetadata.Google = &GoogleProviderMetadata{ThoughtSignature: thoughtSignature}
					}

					if !yield(ToolInputAvailablePart{
						ToolCallID:       toolCallID,
						ToolName:         fc.Name,
						Input:            fc.Args,
						ProviderMetadata: providerMetadata,
					}, nil) {
						return
					}

					finalReason = FinishReasonToolCalls
					continue
				}

				if part.Text == "" {
					continue
				}

				text := part.Text
				if part.Thought {
					if currentContentBlockType != "reasoning" {
						if currentContentBlockType == "text" {
							if !yield(TextEndPart{ID: currentContentBlockIDText}, nil) {
								return
							}
							bumpContentBlockID()
						}
						if !yield(ReasoningStartPart{ID: currentContentBlockIDText}, nil) {
							return
						}
						currentContentBlockType = "reasoning"
					}
					if !yield(ReasoningDeltaPart{
						ID:    currentContentBlockIDText,
						Delta: text,
					}, nil) {
						return
					}
				} else {
					if currentContentBlockType != "text" {
						if currentContentBlockType == "reasoning" {
							if !yield(ReasoningEndPart{ID: currentContentBlockIDText}, nil) {
								return
							}
							bumpContentBlockID()
						}
						if !yield(TextStartPart{ID: currentContentBlockIDText}, nil) {
							return
						}
						currentContentBlockType = "text"
					}
					if !yield(TextDeltaPart{
						ID:    currentContentBlockIDText,
						Delta: text,
					}, nil) {
						return
					}
				}
				// Add handling for other part types (e.g., FunctionResponse) if necessary
			}
		}

		if currentContentBlockType == "text" {
			if !yield(TextEndPart{ID: currentContentBlockIDText}, nil) {
				return
			}
		}
		if currentContentBlockType == "reasoning" {
			if !yield(ReasoningEndPart{ID: currentContentBlockIDText}, nil) {
				return
			}
		}

		// Determine the final reason only *after* the loop completes.
		var actualFinalReason FinishReason
		var finalCand *genai.Candidate

		if lastResp != nil && len(lastResp.Candidates) > 0 {
			finalCand = lastResp.Candidates[0]
		}

		// Use the detected tool call reason if present, otherwise determine from candidate.
		if finalReason == FinishReasonToolCalls {
			actualFinalReason = FinishReasonToolCalls
		} else if finalCand != nil {
			switch finalCand.FinishReason {
			case genai.FinishReasonStop:
				actualFinalReason = FinishReasonStop
			case genai.FinishReasonMaxTokens:
				actualFinalReason = FinishReasonLength
			case genai.FinishReasonSafety:
				actualFinalReason = FinishReasonContentFilter
			case genai.FinishReasonRecitation:
				actualFinalReason = FinishReasonContentFilter // Treat recitation as content filter
			case genai.FinishReasonUnspecified:
				actualFinalReason = FinishReasonUnknown
			default:
				actualFinalReason = FinishReasonOther
			}
		} else {
			// If no candidate and no tool call detected, assume stop or error?
			// Let's default to Stop, assuming the stream ended normally without specific reason.
			actualFinalReason = FinishReasonStop
		}

		if lastResp != nil && lastResp.UsageMetadata != nil {
			usage = lastResp.UsageMetadata
		}

		if !messageStarted {
			if !yield(MessageStartPart{}, nil) {
				return
			}
			if !yield(StartStepStreamPart{}, nil) {
				return
			}
		}

		// Send final finish step part
		if !yield(FinishStepPart{}, nil) {
			return // Stop if yield fails
		}

		// Send final finish message part
		yield(FinishPart{
			FinishReason: actualFinalReason,
		}, nil) // Ignore yield result here as we're at the very end
	}

	return dataStream, getUsage
}
