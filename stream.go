package aisdk

import (
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
)

// Chat is the structure sent from `useChat` to the server.
// This can be extended if you'd like to send additional data with `body`.
type Chat struct {
	ID       string    `json:"id"`
	Messages []Message `json:"messages"`
}

// DataStream is a stream of DataStreamParts.
type DataStream iter.Seq2[DataStreamPart, error]

func SendSingleDataStreamPart(w io.Writer, part DataStreamPart) error {
	messageJson, err := formatJSONPart(part)
	if err != nil {
		return fmt.Errorf("failed to format part: %w", err)
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", messageJson)
	if err != nil {
		return fmt.Errorf("failed to write part to writer: %w", err)
	}
	return nil
}

// Pipe iterates over the DataStream and writes the parts to the writer.
func (s DataStream) Pipe(w io.Writer) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		flusher = nil
	}

	var pipeErr error
	s(func(part DataStreamPart, err error) bool {
		if err != nil {
			errorPart := ErrorPart{ErrorText: err.Error()}
			if messageJson, formatErr := formatJSONPart(errorPart); formatErr == nil {
				_, _ = fmt.Fprintf(w, "data: %s\n\n", messageJson)
				if flusher != nil {
					flusher.Flush()
				}
			}
			pipeErr = err
			return false
		}
		messageJson, err := formatJSONPart(part)
		if err != nil {
			errorPart := ErrorPart{ErrorText: fmt.Sprintf("failed to format part: %s", err.Error())}
			if errorJson, formatErr := formatJSONPart(errorPart); formatErr == nil {
				_, _ = fmt.Fprintf(w, "data: %s\n\n", errorJson)
				if flusher != nil {
					flusher.Flush()
				}
			}
			pipeErr = err
			return false
		}
		_, err = fmt.Fprintf(w, "data: %s\n\n", messageJson)
		if err != nil {
			pipeErr = err
			return false
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	})

	// Send the "[DONE]" termination sequence for v2 protocol
	if pipeErr == nil {
		_, err := fmt.Fprint(w, "[DONE]\n\n")
		if err != nil {
			pipeErr = err
		}
		if flusher != nil {
			flusher.Flush()
		}
	}

	return pipeErr
}

// DataStreamPart represents a part of the Vercel AI SDK data stream.
type DataStreamPart interface {
	Type() string
}

// MessageStartPart indicates the start of a new message.
type MessageStartPart struct {
	MessageID string `json:"messageId,omitempty"`
}

func (p MessageStartPart) Type() string { return "start" }

// TextStartPart indicates the start of a text segment.
type TextStartPart struct {
	ID string `json:"id"`
}

func (p TextStartPart) Type() string { return "text-start" }

// TextDeltaPart contains incremental text content.
type TextDeltaPart struct {
	ID    string `json:"id"`
	Delta string `json:"delta"`
}

func (p TextDeltaPart) Type() string { return "text-delta" }

// TextEndPart indicates the end of a text segment.
type TextEndPart struct {
	ID string `json:"id"`
}

func (p TextEndPart) Type() string { return "text-end" }

// ReasoningStartPart indicates the start of a reasoning segment.
type ReasoningStartPart struct {
	ID string `json:"id"`
}

func (p ReasoningStartPart) Type() string { return "reasoning-start" }

// ReasoningDeltaPart contains incremental reasoning content.
type ReasoningDeltaPart struct {
	ID               string           `json:"id"`
	Delta            string           `json:"delta"`
	ProviderMetadata ProviderMetadata `json:"providerMetadata,omitzero"`
}

func (p ReasoningDeltaPart) Type() string { return "reasoning-delta" }

// ReasoningEndPart indicates the end of a reasoning segment.
type ReasoningEndPart struct {
	ID string `json:"id"`
}

func (p ReasoningEndPart) Type() string { return "reasoning-end" }

// SourceUrlPart provides a URL reference for a source.
type SourceUrlPart struct {
	SourceID string `json:"sourceId"`
	URL      string `json:"url"`
}

func (p SourceUrlPart) Type() string { return "source-url" }

// SourceDocumentPart provides document metadata for a source.
type SourceDocumentPart struct {
	SourceID  string `json:"sourceId"`
	MediaType string `json:"mediaType"`
	Title     string `json:"title"`
}

func (p SourceDocumentPart) Type() string { return "source-document" }

// FilePart provides file content via URL.
type FilePart struct {
	URL       string `json:"url"`
	MediaType string `json:"mediaType"`
}

func (p FilePart) Type() string { return "file" }

// DataPart provides custom data with a type suffix.
type DataPart struct {
	TypeSuffix string `json:"-"` // Not serialized, used to construct type
	Data       any    `json:"data"`
}

func (p DataPart) Type() string { return "data-" + p.TypeSuffix }

// ErrorPart indicates an error occurred.
type ErrorPart struct {
	ErrorText string `json:"errorText"`
}

func (p ErrorPart) Type() string { return "error" }

// ToolCall represents a tool call *request*.
type ToolCall struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type ToolCallResult interface {
	Part | []Part | any
}

// ToolInputStartPart indicates the start of tool input.
type ToolInputStartPart struct {
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
}

func (p ToolInputStartPart) Type() string { return "tool-input-start" }

// ToolInputDeltaPart contains incremental tool input.
type ToolInputDeltaPart struct {
	ToolCallID     string `json:"toolCallId"`
	InputTextDelta string `json:"inputTextDelta"`
}

func (p ToolInputDeltaPart) Type() string { return "tool-input-delta" }

// ToolInputAvailablePart provides complete tool input.
type ToolInputAvailablePart struct {
	ToolCallID       string           `json:"toolCallId"`
	ToolName         string           `json:"toolName"`
	Input            map[string]any   `json:"input"`
	ProviderMetadata ProviderMetadata `json:"providerMetadata,omitzero"`
}

func (p ToolInputAvailablePart) Type() string { return "tool-input-available" }

// ToolOutputAvailablePart provides tool output.
type ToolOutputAvailablePart struct {
	ToolCallID string `json:"toolCallId"`
	Output     any    `json:"output"`
}

func (p ToolOutputAvailablePart) Type() string { return "tool-output-available" }

// StartStepStreamPart corresponds to TYPE_ID 'f'.
type StartStepStreamPart struct {
}

func (p StartStepStreamPart) Type() string { return "start-step" }

// FinishReason defines the possible reasons for finishing a step or message.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content-filter"
	FinishReasonToolCalls     FinishReason = "tool-calls"
	FinishReasonError         FinishReason = "error"
	FinishReasonOther         FinishReason = "other"
	FinishReasonUnknown       FinishReason = "unknown"
)

// Usage details the token usage.
type Usage struct {
	PromptTokens     *int64 `json:"promptTokens"`
	CompletionTokens *int64 `json:"completionTokens"`
}

// FinishStepPart indicates the completion of a step.
type FinishStepPart struct{}

func (p FinishStepPart) Type() string { return "finish-step" }

// FinishPart indicates the completion of a message.
type FinishPart struct {
	FinishReason FinishReason `json:"finishReason,omitzero"`
}

func (p FinishPart) Type() string { return "finish" }

func formatJSONPart(part DataStreamPart) (string, error) {
	jsonData, err := json.Marshal(part)
	if err != nil {
		return "", fmt.Errorf("failed to marshal part type %T: %w", part, err)
	}
	var document map[string]any
	if err := json.Unmarshal(jsonData, &document); err != nil {
		return "", fmt.Errorf("failed to unmarshal JSON for part type %T: %w", part, err)
	}
	document["type"] = part.Type()
	finalData, err := json.Marshal(document)
	if err != nil {
		return "", fmt.Errorf("failed to marshal final JSON for part type %T: %w", part, err)
	}

	return string(finalData), nil
}

type Attachment struct {
	Name        string `json:"name,omitempty"`
	ContentType string `json:"contentType,omitempty"`
	URL         string `json:"url"`
}

type Message struct {
	ID          string           `json:"id"`
	CreatedAt   *json.RawMessage `json:"createdAt,omitempty"`
	Content     string           `json:"content"`
	Role        string           `json:"role"`
	Parts       []Part           `json:"parts,omitempty"`
	Annotations []any            `json:"annotations,omitempty"`
	Attachments []Attachment     `json:"experimental_attachments,omitempty"`
}

type PartType string

const (
	PartTypeText           PartType = "text"
	PartTypeReasoning      PartType = "reasoning"
	PartTypeToolInvocation PartType = "tool-invocation"
	PartTypeSource         PartType = "source"
	PartTypeFile           PartType = "file"
	PartTypeStepStart      PartType = "step-start"
)

type ReasoningDetail struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
}

type SourceInfo struct {
	URI         string         `json:"uri,omitempty"`
	ContentType string         `json:"contentType,omitempty"`
	Data        string         `json:"data,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

type Part struct {
	Type PartType `json:"type"`

	// Type: "text"
	Text string `json:"text,omitempty"`

	// Type: "reasoning"
	Reasoning string            `json:"reasoning,omitempty"`
	Details   []ReasoningDetail `json:"details,omitempty"`

	// Type: "tool-invocation"
	ToolCallID string              `json:"toolCallId"`
	ToolName   string              `json:"toolName"`
	Input      any                 `json:"input"`
	Output     any                 `json:"output,omitempty"`
	State      ToolInvocationState `json:"state,omitempty"`
	ErrorText  string              `json:"errorText,omitempty"`

	// Type: "source"
	Source *SourceInfo `json:"source,omitempty"`

	// Type: "file"
	MimeType string `json:"mimeType,omitempty"`
	Data     []byte `json:"data,omitempty"`

	// Type: "step-start" - No additional fields

	ProviderMetadata *ProviderMetadata `json:"providerMetadata,omitzero"`
}

type ProviderMetadata struct {
	Anthropic *AnthropicProviderMetadata `json:"anthropic,omitzero"`
	Google    *GoogleProviderMetadata    `json:"google,omitzero"`
}

type AnthropicProviderMetadata struct {
	Signature string `json:"signature,omitempty"` // Optional signature for reasoning
}

type GoogleProviderMetadata struct {
	ThoughtSignature []byte `json:"thoughtSignature,omitempty"` // Thought signature for function calls (Gemini 3+)
}

func (p *Part) UnmarshalJSON(data []byte) error {
	var justTheType struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &justTheType); err != nil {
		return fmt.Errorf("failed to unmarshal part type: %w", err)
	}

	type alias Part
	temp := struct {
		*alias
		// CallProviderMetadata seems to be sent for tool-invocation parts wheras
		// all the other parts use regular ProviderMetadata. So this is a workaround
		// to unmarshal both cases into the same field.
		CallProviderMetadata *ProviderMetadata `json:"callProviderMetadata"`
	}{alias: (*alias)(p)}

	if err := json.Unmarshal(data, &temp); err != nil {
		return fmt.Errorf("failed to unmarshal part normally: %w", err)
	}

	if strings.HasPrefix(justTheType.Type, "tool-") {
		p.Type = PartTypeToolInvocation
		p.ToolName = strings.TrimPrefix(justTheType.Type, "tool-")
		p.ProviderMetadata = temp.CallProviderMetadata
	}

	return nil
}

type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Schema      Schema `json:"schema"`
}

type Schema struct {
	Required   []string       `json:"required"`
	Properties map[string]any `json:"properties"`
}

type ToolInvocationState string

const (
	ToolInvocationStateCall            ToolInvocationState = "call"
	ToolInvocationStatePartialCall     ToolInvocationState = "partial-call"
	ToolInvocationStateResult          ToolInvocationState = "result"
	ToolInvocationStateOutputAvailable ToolInvocationState = "output-available"
	ToolInvocationStateOutputError     ToolInvocationState = "output-error"
)

func WriteDataStreamHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("X-Vercel-AI-Data-Stream", "v2")
	w.Header().Set("X-Accel-Buffering", "no") // Disable buffering for real-time streaming
	w.WriteHeader(http.StatusOK)
}

func toolResultToParts(result any) ([]Part, error) {
	switch r := result.(type) {
	case []Part:
		return r, nil
	case Part:
		return []Part{r}, nil
	default:
		jsonData, err := json.Marshal(r)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal tool call result: %w", err)
		}
		return []Part{{Type: PartTypeText, Text: string(jsonData)}}, nil
	}
}
