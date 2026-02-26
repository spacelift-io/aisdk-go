package aisdk

import (
	"encoding/base64"
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

// StreamTransformer transforms stream parts as they flow through.
// It can modify or replace parts before they reach the consumer.
type StreamTransformer func(DataStreamPart) DataStreamPart

// StreamProcessor observes stream parts as they flow through.
// Processors cannot modify parts, only observe them.
type StreamProcessor func(DataStreamPart)

// WithTransformers wraps the DataStream to transform parts before they reach the consumer.
func (s DataStream) WithTransformers(transformers ...StreamTransformer) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		s(func(part DataStreamPart, err error) bool {
			if err == nil && part != nil {
				for _, t := range transformers {
					part = t(part)
				}
			}
			return yield(part, err)
		})
	}
}

// WithProcessors wraps the DataStream to call processors for each part.
// Processors observe parts but cannot modify them.
func (s DataStream) WithProcessors(processors ...StreamProcessor) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		s(func(part DataStreamPart, err error) bool {
			if err == nil && part != nil {
				for _, p := range processors {
					p(part)
				}
			}
			return yield(part, err)
		})
	}
}

// ReplyToMessageID returns a transformer that sets the message ID on MessageStartPart.
// Use this when continuing/updating an existing message so useChat updates it in place.
func ReplyToMessageID(messageID string) StreamTransformer {
	return func(part DataStreamPart) DataStreamPart {
		if p, ok := part.(MessageStartPart); ok {
			p.MessageID = messageID
			return p
		}
		return part
	}
}

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

func WriteDone(w io.Writer) error {
	_, err := fmt.Fprint(w, "[DONE]\n\n")
	return err
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

func (p Part) MarshalJSON() ([]byte, error) {
	switch p.Type {
	case PartTypeText:
		data := map[string]any{"type": "text", "text": p.Text}
		if p.State != "" {
			data["state"] = string(p.State)
		}
		if p.ProviderMetadata != nil {
			data["providerMetadata"] = p.ProviderMetadata
		}
		return json.Marshal(data)

	case PartTypeReasoning:
		data := map[string]any{"type": "reasoning", "text": p.Reasoning}
		if p.State != "" {
			data["state"] = string(p.State)
		}
		if len(p.Details) > 0 {
			data["details"] = p.Details
		}
		if p.ProviderMetadata != nil {
			data["providerMetadata"] = p.ProviderMetadata
		}
		return json.Marshal(data)

	case PartTypeToolInvocation:
		data := map[string]any{
			"type":       "tool-" + p.ToolName,
			"toolCallId": p.ToolCallID,
		}
		if p.State != "" {
			data["state"] = string(p.State)
		}
		if p.Input != nil {
			data["input"] = p.Input
		}
		if p.Output != nil {
			data["output"] = p.Output
		}
		if p.ErrorText != "" {
			data["errorText"] = p.ErrorText
		}
		if p.ProviderMetadata != nil {
			data["callProviderMetadata"] = p.ProviderMetadata
		}
		return json.Marshal(data)

	case PartTypeFile:
		data := map[string]any{
			"type":      "file",
			"mediaType": p.MimeType,
		}
		if len(p.Data) > 0 {
			data["url"] = "data:" + p.MimeType + ";base64," + base64.StdEncoding.EncodeToString(p.Data)
		}
		if p.ProviderMetadata != nil {
			data["providerMetadata"] = p.ProviderMetadata
		}
		return json.Marshal(data)

	case PartTypeStepStart:
		return json.Marshal(map[string]string{"type": "step-start"})

	case PartTypeSource:
		data := map[string]any{"type": "source"}
		if p.Source != nil {
			data["source"] = p.Source
		}
		if p.ProviderMetadata != nil {
			data["providerMetadata"] = p.ProviderMetadata
		}
		return json.Marshal(data)

	default:
		type alias Part
		return json.Marshal(struct {
			alias
			Type string `json:"type"`
		}{alias: alias(p), Type: string(p.Type)})
	}
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
	// useChat tool states
	ToolStateInputStreaming  ToolInvocationState = "input-streaming"
	ToolStateInputAvailable  ToolInvocationState = "input-available"
	ToolStateOutputAvailable ToolInvocationState = "output-available"
	ToolStateOutputError     ToolInvocationState = "output-error"

	// Text/reasoning states (same type, different values)
	StateStreaming ToolInvocationState = "streaming"
	StateDone      ToolInvocationState = "done"

	// Legacy states for backward compatibility
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

// MessageCollector accumulates stream parts into a Message compatible with useChat round-trip.
type MessageCollector struct {
	message              Message
	activeTextParts      map[string]*partAccumulator
	activeReasoningParts map[string]*partAccumulator
	activeToolParts      map[string]int // toolCallId -> index in Parts
}

type partAccumulator struct {
	index  int
	buffer strings.Builder
}

// NewMessageCollector creates a new MessageCollector for accumulating stream parts.
func NewMessageCollector() *MessageCollector {
	return &MessageCollector{
		message:              Message{Role: "assistant"},
		activeTextParts:      make(map[string]*partAccumulator),
		activeReasoningParts: make(map[string]*partAccumulator),
		activeToolParts:      make(map[string]int),
	}
}

// Process implements StreamProcessor to accumulate parts into a Message.
func (c *MessageCollector) Process(part DataStreamPart) {
	switch p := part.(type) {
	case MessageStartPart:
		c.message.ID = p.MessageID

	// --- Text ---
	case TextStartPart:
		idx := len(c.message.Parts)
		c.message.Parts = append(c.message.Parts, Part{
			Type:  PartTypeText,
			State: StateStreaming,
		})
		c.activeTextParts[p.ID] = &partAccumulator{index: idx}

	case TextDeltaPart:
		if acc, ok := c.activeTextParts[p.ID]; ok {
			acc.buffer.WriteString(p.Delta)
			c.message.Parts[acc.index].Text = acc.buffer.String()
		}

	case TextEndPart:
		if acc, ok := c.activeTextParts[p.ID]; ok {
			c.message.Parts[acc.index].Text = acc.buffer.String()
			c.message.Parts[acc.index].State = StateDone
			delete(c.activeTextParts, p.ID)
		}

	// --- Reasoning ---
	case ReasoningStartPart:
		idx := len(c.message.Parts)
		c.message.Parts = append(c.message.Parts, Part{
			Type:  PartTypeReasoning,
			State: StateStreaming,
		})
		c.activeReasoningParts[p.ID] = &partAccumulator{index: idx}

	case ReasoningDeltaPart:
		if acc, ok := c.activeReasoningParts[p.ID]; ok {
			acc.buffer.WriteString(p.Delta)
			c.message.Parts[acc.index].Reasoning = acc.buffer.String()
			if p.ProviderMetadata.Anthropic != nil {
				c.message.Parts[acc.index].ProviderMetadata = &p.ProviderMetadata
			}
		}

	case ReasoningEndPart:
		if acc, ok := c.activeReasoningParts[p.ID]; ok {
			c.message.Parts[acc.index].Reasoning = acc.buffer.String()
			c.message.Parts[acc.index].State = StateDone
			delete(c.activeReasoningParts, p.ID)
		}

	// --- Tools ---
	case ToolInputStartPart:
		idx := len(c.message.Parts)
		c.message.Parts = append(c.message.Parts, Part{
			Type:       PartTypeToolInvocation,
			ToolCallID: p.ToolCallID,
			ToolName:   p.ToolName,
			State:      ToolStateInputStreaming,
		})
		c.activeToolParts[p.ToolCallID] = idx

	case ToolInputAvailablePart:
		idx, ok := c.activeToolParts[p.ToolCallID]
		if !ok {
			idx = len(c.message.Parts)
			c.message.Parts = append(c.message.Parts, Part{
				Type:       PartTypeToolInvocation,
				ToolCallID: p.ToolCallID,
				ToolName:   p.ToolName,
			})
			c.activeToolParts[p.ToolCallID] = idx
		}
		c.message.Parts[idx].Input = p.Input
		c.message.Parts[idx].State = ToolStateInputAvailable
		if p.ProviderMetadata.Anthropic != nil || p.ProviderMetadata.Google != nil {
			c.message.Parts[idx].ProviderMetadata = &p.ProviderMetadata
		}

	case ToolOutputAvailablePart:
		if idx, ok := c.activeToolParts[p.ToolCallID]; ok {
			c.message.Parts[idx].Output = p.Output
			c.message.Parts[idx].State = ToolStateOutputAvailable
		}

	// --- Steps ---
	case FinishStepPart:
		c.activeTextParts = make(map[string]*partAccumulator)
		c.activeReasoningParts = make(map[string]*partAccumulator)
	}
}

// Message returns the accumulated Message.
func (c *MessageCollector) Message() Message {
	return c.message
}

// MergeParts merges incoming parts with existing parts.
// Tool invocation parts are merged by ToolCallID (updating existing ones).
// Step-start parts are filtered out. Other parts are deduplicated by content.
func MergeParts(existing, incoming []Part) []Part {
	result := make([]Part, 0, len(existing)+len(incoming))
	toolPartIdx := make(map[string]int)

	for _, p := range existing {
		if p.Type == PartTypeStepStart {
			continue
		}
		if p.Type == PartTypeToolInvocation && p.ToolCallID != "" {
			toolPartIdx[p.ToolCallID] = len(result)
		}
		result = append(result, p)
	}

	for _, p := range incoming {
		if p.Type == PartTypeStepStart {
			continue
		}

		if p.Type == PartTypeToolInvocation && p.ToolCallID != "" {
			if idx, ok := toolPartIdx[p.ToolCallID]; ok {
				result[idx] = p
				continue
			}
			toolPartIdx[p.ToolCallID] = len(result)
			result = append(result, p)
			continue
		}

		if !containsSimilarPart(result, p) {
			result = append(result, p)
		}
	}

	return result
}

func containsSimilarPart(parts []Part, p Part) bool {
	for _, existing := range parts {
		if existing.Type != p.Type {
			continue
		}
		switch p.Type {
		case PartTypeText:
			if existing.Text == p.Text {
				return true
			}
		case PartTypeReasoning:
			if existing.Reasoning == p.Reasoning {
				return true
			}
		}
	}
	return false
}
