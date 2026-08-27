import { useCallback, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { useSpeechRecognition } from "@/hooks/useSpeechRecognition";
import "./SearchBox.css";

type SearchBoxProps = {
  isGenerating: boolean;
  sendMessage: (content: string) => Promise<void>;
  stopMessage: () => void;
};

const SearchBox = ({ isGenerating, sendMessage, stopMessage }: SearchBoxProps) => {
  const [inputValue, setInputValue] = useState("");

  const handleTranscript = useCallback((transcript: string) => {
    setInputValue((currentValue) => {
      if (!currentValue.trim()) return transcript;
      return `${currentValue.trimEnd()} ${transcript}`;
    });
  }, []);

  const {
    error: voiceError,
    isSupported: isVoiceSupported,
    status: voiceStatus,
    transcript: voiceTranscript,
    toggle: toggleVoiceInput,
  } = useSpeechRecognition({ onTranscript: handleTranscript });

  const handleSearchSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const content = inputValue.trim();
    if (!content || isGenerating) return;

    void sendMessage(content);
    setInputValue("");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter") return;
    if (event.nativeEvent.isComposing) return;
    if (event.shiftKey) return;

    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  };

  const voiceButtonLabel = voiceStatus === "recording"
    ? "停止语音输入"
    : "开始语音输入";

  return (
    <div className="searchBoxWrapper">
      <form className="searchBox" onSubmit={handleSearchSubmit}>
        <textarea
          rows={1}
          aria-label="消息内容"
          placeholder="请输入问题"
          value={voiceStatus === "recording" ? voiceTranscript || inputValue : inputValue}
          readOnly={voiceStatus !== "idle"}
          onKeyDown={handleKeyDown}
          onChange={(event) => setInputValue(event.target.value)}
        />

        <div className="searchActions">
          <button
            type="button"
            className={`voiceButton ${voiceStatus}`}
            aria-label={voiceButtonLabel}
            aria-pressed={voiceStatus === "recording"}
            title={isVoiceSupported ? voiceButtonLabel : "当前浏览器不支持语音输入"}
            disabled={!isVoiceSupported || isGenerating || voiceStatus === "processing"}
            onClick={toggleVoiceInput}
          >
            <span aria-hidden="true">🎙</span>
          </button>

          {isGenerating ? (
            <button type="button" className="stopButton" onClick={() => stopMessage()}>
              停止
            </button>
          ) : (
            <button type="submit" disabled={!inputValue.trim()}>
              发送
            </button>
          )}
        </div>
      </form>

      <div className="voiceFeedback" aria-live="polite">
        {voiceError ? (
          <p className="voiceError" role="alert">{voiceError}</p>
        ) : voiceStatus !== "idle" ? (
          <p>{voiceStatus === "recording" ? "正在聆听，再次点击麦克风结束" : "正在整理识别结果"}</p>
        ) : null}
      </div>
    </div>
  );
};

export default SearchBox;
