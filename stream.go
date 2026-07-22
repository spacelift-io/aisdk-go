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

// StreamTransformer transforms stream parts as they flow through. For each
// part it returns the parts to emit in its place: one part to rewrite, several
// to expand, an empty non-nil slice to drop, or nil to pass the original
// through unchanged.
type StreamTransformer func(DataStreamPart) []DataStreamPart

// StreamProcessor observes stream parts as they flow through.
// Processors cannot modify parts, only observe them.
type StreamProcessor func(DataStreamPart)

// WithTransformers wraps the DataStream to transform parts before they reach
// the consumer. Transformers apply left to right; parts produced by one
// transformer flow through the remaining ones.
func (s DataStream) WithTransformers(transformers ...StreamTransformer) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		s(func(part DataStreamPart, err error) bool {
			if err != nil || part == nil {
				return yield(part, err)
			}

			parts := []DataStreamPart{part}
			for _, t := range transformers {
				next := make([]DataStreamPart, 0, len(parts))
				for _, p := range parts {
					if out := t(p); out == nil {
						next = append(next, p)
					} else {
						next = append(next, out...)
					}
				}
				parts = next
			}

			for _, p := range parts {
				if !yield(p, nil) {
					return false
				}
			}
			return true
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

// InjectToolApprovalRequests returns a transformer that emits a
// tool-approval-request part after each tool-input-available part whose tool
// requires user approval. Providers know nothing about approvals, so the
// request has to be fabricated between the provider stream and the client.
func InjectToolApprovalRequests(needsApproval func(toolName string, input map[string]any) bool, newApprovalID func() string) StreamTransformer {
	return func(part DataStreamPart) []DataStreamPart {
		p, ok := part.(ToolInputAvailablePart)
		if !ok || !needsApproval(p.ToolName, p.Input) {
			return nil
		}
		return []DataStreamPart{
			p,
			ToolApprovalRequestPart{
				ApprovalID: newApprovalID(),
				ToolCallID: p.ToolCallID,
			},
		}
	}
}

// MarkDeniedToolCalls processes tool call denials that arrived from the
// client.
//
// When the user denies a tool call, useChat records the decision on the tool
// part (state approval-responded with approval.approved false) and resends
// the message. That state means answered but not yet processed: the part has
// no output, and the client keeps it non-terminal until the server
// acknowledges the denial with a tool-output-denied stream part.
//
// This function finds such parts, moves them to their terminal output-denied
// state, and returns their tool call IDs so the caller can emit the
// acknowledgment on the outgoing stream with [InjectToolOutputsDenied].
//
// Call it on incoming messages before persisting them and before building
// provider prompts. Persisting the flipped state makes reloads serve the same
// terminal state the live stream produced, and the flip doubles as the
// exactly-once guard: a retried request finds output-denied instead of
// approval-responded and returns nothing.
func MarkDeniedToolCalls(messages []Message) []string {
	var toolCallIDs []string
	for mi := range messages {
		if messages[mi].Role != "assistant" {
			continue
		}
		for pi := range messages[mi].Parts {
			part := &messages[mi].Parts[pi]
			if part.Type != PartTypeToolInvocation ||
				part.State != ToolStateApprovalResponded ||
				part.Approval == nil ||
				part.Approval.Approved == nil ||
				*part.Approval.Approved {
				continue
			}
			part.State = ToolStateOutputDenied
			toolCallIDs = append(toolCallIDs, part.ToolCallID)
		}
	}
	return toolCallIDs
}

// InjectToolOutputsDenied returns a transformer that emits a
// tool-output-denied part per tool call ID right after the message start, so
// the client moves the denied tool parts of the replied-to message into their
// terminal state before new content arrives. Pair it with
// [MarkDeniedToolCalls] on the incoming messages.
func InjectToolOutputsDenied(toolCallIDs ...string) StreamTransformer {
	injected := false
	return func(part DataStreamPart) []DataStreamPart {
		if injected || len(toolCallIDs) == 0 {
			return nil
		}
		p, ok := part.(MessageStartPart)
		if !ok {
			return nil
		}
		injected = true
		parts := make([]DataStreamPart, 0, len(toolCallIDs)+1)
		parts = append(parts, p)
		for _, toolCallID := range toolCallIDs {
			parts = append(parts, ToolOutputDeniedPart{ToolCallID: toolCallID})
		}
		return parts
	}
}

// NeedsApprovalFromTools returns a predicate for [InjectToolApprovalRequests]
// based on the NeedsApproval flag of the provided tool definitions. The input
// argument is ignored; it exists so callers can substitute input-dependent
// policies without changing the transformer.
func NeedsApprovalFromTools(tools []Tool) func(toolName string, input map[string]any) bool {
	names := make(map[string]bool, len(tools))
	for _, tool := range tools {
		if tool.NeedsApproval {
			names[tool.Name] = true
		}
	}
	return func(toolName string, _ map[string]any) bool {
		return names[toolName]
	}
}

// ReplyToMessageID returns a transformer that sets the message ID on MessageStartPart.
// Use this when continuing/updating an existing message so useChat updates it in place.
func ReplyToMessageID(messageID string) StreamTransformer {
	return func(part DataStreamPart) []DataStreamPart {
		if p, ok := part.(MessageStartPart); ok {
			p.MessageID = messageID
			return []DataStreamPart{p}
		}
		return nil
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
	ID               string           `json:"id"`
	Delta            string           `json:"delta"`
	ProviderMetadata ProviderMetadata `json:"providerMetadata,omitzero"`
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
	Title      string `json:"title,omitempty"`
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
	Title            string           `json:"title,omitempty"`
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

// ToolApprovalRequestPart asks the client for permission to run a tool call.
// IsAutomatic and Signature exist in the v7 protocol only; ai@6 clients
// validate stream parts strictly, so leave them unset until the frontend
// upgrades.
type ToolApprovalRequestPart struct {
	ApprovalID  string `json:"approvalId"`
	ToolCallID  string `json:"toolCallId"`
	IsAutomatic bool   `json:"isAutomatic,omitempty"`
	Signature   string `json:"signature,omitempty"`
}

func (p ToolApprovalRequestPart) Type() string { return "tool-approval-request" }

// ToolApprovalResponsePart records an approval decision on the stream. The
// v6 protocol has no such part (decisions travel back inside message parts);
// it is only ever streamed for server-decided automatic approvals on v7+.
type ToolApprovalResponsePart struct {
	ApprovalID       string `json:"approvalId"`
	Approved         bool   `json:"approved"`
	Reason           string `json:"reason,omitempty"`
	ProviderExecuted bool   `json:"providerExecuted,omitempty"`
}

func (p ToolApprovalResponsePart) Type() string { return "tool-approval-response" }

// ToolOutputDeniedPart moves a denied tool call to its terminal state on the
// client after the server has processed the denial.
type ToolOutputDeniedPart struct {
	ToolCallID string `json:"toolCallId"`
}

func (p ToolOutputDeniedPart) Type() string { return "tool-output-denied" }

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
	Title      string              `json:"title,omitempty"`
	Input      any                 `json:"input"`
	Output     any                 `json:"output,omitempty"`
	State      ToolInvocationState `json:"state,omitempty"`
	ErrorText  string              `json:"errorText,omitempty"`
	Approval   *ToolApproval       `json:"approval,omitempty"`

	// Type: "source"
	Source *SourceInfo `json:"source,omitempty"`

	// Type: "file"
	MimeType string `json:"mimeType,omitempty"`
	Data     []byte `json:"data,omitempty"`

	// Type: "step-start" - No additional fields

	ProviderMetadata *ProviderMetadata `json:"providerMetadata,omitzero"`
}

// ToolApproval carries the approval lifecycle of a tool-invocation part, as
// exchanged with useChat inside message parts. Approved stays nil while the
// request awaits a decision. Signature is passed through verbatim; this
// library performs no verification.
type ToolApproval struct {
	ID          string `json:"id"`
	Approved    *bool  `json:"approved,omitempty"`
	Reason      string `json:"reason,omitempty"`
	IsAutomatic bool   `json:"isAutomatic,omitempty"`
	Signature   string `json:"signature,omitempty"`
}

type ProviderMetadata struct {
	Anthropic *AnthropicProviderMetadata `json:"anthropic,omitzero"`
	Google    *GoogleProviderMetadata    `json:"google,omitzero"`
	Bedrock   *BedrockProviderMetadata   `json:"bedrock,omitzero"`
}

type AnthropicProviderMetadata struct {
	Signature    string `json:"signature,omitempty"`    // Optional signature for reasoning
	RedactedData string `json:"redactedData,omitempty"` // Encrypted payload of a redacted_thinking block, replayed verbatim
}

type GoogleProviderMetadata struct {
	ThoughtSignature []byte `json:"thoughtSignature,omitempty"` // Thought signature for function calls (Gemini 3+)
}

type BedrockProviderMetadata struct {
	Signature    string `json:"signature,omitempty"`    // Signature for Bedrock reasoning replay
	RedactedData string `json:"redactedData,omitempty"` // Base64-encoded redacted reasoning payload
}

func hasProviderMetadata(metadata ProviderMetadata) bool {
	return metadata.Anthropic != nil || metadata.Google != nil || metadata.Bedrock != nil
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
		if p.Title != "" {
			data["title"] = p.Title
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
		if p.Approval != nil {
			data["approval"] = p.Approval
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
	Name          string `json:"name"`
	Description   string `json:"description"`
	Title         string `json:"title,omitempty"`
	NeedsApproval bool   `json:"needsApproval,omitempty"`
	Schema        Schema `json:"schema"`
}

// InjectToolTitlesFromTools returns a transformer that injects tool titles
// into streamed tool input parts based on the provided tool definitions.
//
// It only sets title when the stream part does not already include one.
func InjectToolTitlesFromTools(tools []Tool) StreamTransformer {
	titlesByName := make(map[string]string, len(tools))
	for _, tool := range tools {
		if tool.Title == "" {
			continue
		}
		// Last non-empty title wins for duplicate tool names.
		titlesByName[tool.Name] = tool.Title
	}

	if len(titlesByName) == 0 {
		return func(DataStreamPart) []DataStreamPart { return nil }
	}

	return func(part DataStreamPart) []DataStreamPart {
		switch p := part.(type) {
		case ToolInputStartPart:
			if p.Title == "" {
				p.Title = titlesByName[p.ToolName]
			}
			return []DataStreamPart{p}
		case ToolInputAvailablePart:
			if p.Title == "" {
				p.Title = titlesByName[p.ToolName]
			}
			return []DataStreamPart{p}
		default:
			return nil
		}
	}
}

type Schema struct {
	Required   []string       `json:"required"`
	Properties map[string]any `json:"properties"`
}

type ToolInvocationState string

const (
	// useChat tool states
	ToolStateInputStreaming    ToolInvocationState = "input-streaming"
	ToolStateInputAvailable    ToolInvocationState = "input-available"
	ToolStateApprovalRequested ToolInvocationState = "approval-requested"
	ToolStateApprovalResponded ToolInvocationState = "approval-responded"
	ToolStateOutputAvailable   ToolInvocationState = "output-available"
	ToolStateOutputError       ToolInvocationState = "output-error"
	ToolStateOutputDenied      ToolInvocationState = "output-denied"

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

// deniedToolResultText matches the wording the Vercel AI SDK uses when a
// denied tool call is surfaced to the model as a tool result.
const deniedToolResultText = "Tool call execution denied."

// isDeniedToolPart reports whether the tool-invocation part represents a
// denied tool call: either already in its terminal output-denied state, or
// carrying a negative approval decision that has not been processed yet.
func isDeniedToolPart(p Part) bool {
	if p.Type != PartTypeToolInvocation {
		return false
	}
	if p.State == ToolStateOutputDenied {
		return true
	}
	return p.State == ToolStateApprovalResponded && p.Approval != nil &&
		p.Approval.Approved != nil && !*p.Approval.Approved
}

func deniedToolResultReason(p Part) string {
	if p.Approval != nil && p.Approval.Reason != "" {
		return p.Approval.Reason
	}
	return deniedToolResultText
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
	activeToolParts      map[string]int    // toolCallId -> index in Parts
	approvalToToolCall   map[string]string // approvalId -> toolCallId
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
		approvalToToolCall:   make(map[string]string),
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
			if hasProviderMetadata(p.ProviderMetadata) {
				c.message.Parts[acc.index].ProviderMetadata = &p.ProviderMetadata
			}
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
			if hasProviderMetadata(p.ProviderMetadata) {
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
			Title:      p.Title,
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
				Title:      p.Title,
			})
			c.activeToolParts[p.ToolCallID] = idx
		}
		if p.Title != "" {
			c.message.Parts[idx].Title = p.Title
		}
		c.message.Parts[idx].Input = p.Input
		c.message.Parts[idx].State = ToolStateInputAvailable
		if hasProviderMetadata(p.ProviderMetadata) {
			c.message.Parts[idx].ProviderMetadata = &p.ProviderMetadata
		}

	case ToolOutputAvailablePart:
		if idx, ok := c.activeToolParts[p.ToolCallID]; ok {
			c.message.Parts[idx].Output = p.Output
			c.message.Parts[idx].State = ToolStateOutputAvailable
		}

	case ToolApprovalRequestPart:
		if idx, ok := c.activeToolParts[p.ToolCallID]; ok {
			c.message.Parts[idx].State = ToolStateApprovalRequested
			c.message.Parts[idx].Approval = &ToolApproval{
				ID:          p.ApprovalID,
				IsAutomatic: p.IsAutomatic,
				Signature:   p.Signature,
			}
			c.approvalToToolCall[p.ApprovalID] = p.ToolCallID
		}

	case ToolApprovalResponsePart:
		toolCallID, ok := c.approvalToToolCall[p.ApprovalID]
		if !ok {
			break
		}
		if idx, ok := c.activeToolParts[toolCallID]; ok {
			approved := p.Approved
			approval := c.message.Parts[idx].Approval
			if approval == nil {
				approval = &ToolApproval{ID: p.ApprovalID}
				c.message.Parts[idx].Approval = approval
			}
			approval.Approved = &approved
			approval.Reason = p.Reason
			c.message.Parts[idx].State = ToolStateApprovalResponded
		}

	case ToolOutputDeniedPart:
		if idx, ok := c.activeToolParts[p.ToolCallID]; ok {
			c.message.Parts[idx].State = ToolStateOutputDenied
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
