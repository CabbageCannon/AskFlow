import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// 语音输入三种状态：idle空闲、recording正在录音、processing用户已停止录音
export type VoiceInputStatus = "idle" | "recording" | "processing";

// 表示一条语音的候选文本
type SpeechAlternativeLike = {
  transcript: string;
};

// 一条语音的识别结果
type SpeechResultLike = {
  // 结果是否是最终结果
  readonly isFinal: boolean;
  readonly length: number;
  readonly [index: number]: SpeechAlternativeLike;
};

// 表示一组识别结果列表
type SpeechResultListLike = {
  readonly length: number;
  readonly [index: number]: SpeechResultLike;
};

// 语音识别onresult的事件类型
type SpeechResultEventLike = Event & {
  resultIndex: number;
  results: SpeechResultListLike;
};

// 语音识别错误事件类型
type SpeechErrorEventLike = Event & {
  error: string;
};

// 描述浏览器的语音识别对象
type SpeechRecognitionLike = {
  lang: string; //识别语言
  continuous: boolean; //是否连续识别
  interimResults: boolean; //是否返回临时识别结果
  maxAlternatives: number; //最多返回几个候选识别结果
  onresult: ((event: SpeechResultEventLike) => void) | null; //识别出问题时触发
  onerror: ((event: SpeechErrorEventLike) => void) | null; //识别出错时触发
  onend: (() => void) | null; //识别结束时触发
  start: () => void; //开始识别
  stop: () => void; //结束识别
  abort: () => void; //直接中断识别过程
}; 

// 用于创建新的语音识别实例
type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

type SpeechWindow = Window & {
  SpeechRecognition?: SpeechRecognitionConstructor;
  webkitSpeechRecognition?: SpeechRecognitionConstructor;
};

type UseSpeechRecognitionOptions = {
  onTranscript: (transcript: string) => void;
};

const ERROR_MESSAGES: Record<string, string> = {
  "audio-capture": "无法访问麦克风，请检查设备是否可用。",
  network: "语音识别网络异常，请稍后重试。",
  "no-speech": "没有检测到语音，请再试一次。",
  "not-allowed": "麦克风权限被拒绝，请在浏览器设置中允许访问。",
  "service-not-allowed": "当前环境不允许使用语音识别服务。",
};

const getSpeechRecognition = (): SpeechRecognitionConstructor | null => {
  if (typeof window === "undefined") return null;

  const speechWindow = window as SpeechWindow;
  return speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition ?? null;
};

export const useSpeechRecognition = ({ onTranscript }: UseSpeechRecognitionOptions) => {
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  // 保存最新的识别结果
  const latestTranscriptRef = useRef("");
  // 表示这次识别是不是被主动取消了(有时候识别中断可能不是用户直接造成的,比如组件卸载当前hook,这种时候不应该认为是用户中断识别)
  const cancelledRef = useRef(false);
  // 表示识别过程是否已经出错
  const recognitionFailedRef = useRef(false);

  const [status, setStatus] = useState<VoiceInputStatus>("idle");
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");

  const isSupported = useMemo(() => Boolean(getSpeechRecognition()), []);

  const start = useCallback(() => {
    if (!isSupported) {
      setError("当前浏览器不支持语音输入。");
      return;
    }

    // 如果当前不是空闲状态、或者没有拿到语音识别实例，直接返回
    if (!recognitionRef.current || status !== "idle") return;

    cancelledRef.current = false;
    recognitionFailedRef.current = false;
    latestTranscriptRef.current = "";
    setTranscript("");
    setError("");
    setStatus("recording");

    try {
      recognitionRef.current.start();
    } catch {
      setStatus("idle");
      setError("语音识别启动失败，请稍后重试。");
    }
  }, [isSupported, status]);

  const stop = useCallback(() => {
    // 如果当前没有语音识别实例或者并不是正在录音的状态，直接返回
    if (!recognitionRef.current || status !== "recording") return;

    setStatus("processing");
    recognitionRef.current.stop();
  }, [status]);

  const toggle = useCallback(() => {
    if (status === "recording") {
      stop();
      return;
    }

    if (status === "idle") start();
  }, [start, status, stop]);

  useEffect(() => {
    const SpeechRecognition = getSpeechRecognition();
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.lang = "zh-CN";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      let nextTranscript = "";

      for (let index = 0; index < event.results.length; index += 1) {
        nextTranscript += event.results[index]?.[0]?.transcript ?? "";
      }

      latestTranscriptRef.current = nextTranscript;
      setTranscript(nextTranscript);
    };

    recognition.onerror = (event) => {
      // 如果是用户主动取消的，就不应该表示错误
      if (cancelledRef.current && event.error === "aborted") return;

      recognitionFailedRef.current = true;
      setError(ERROR_MESSAGES[event.error] ?? "语音识别被中断，请稍后重试。");
      setStatus("idle");
    };

    // 无论是正常结束、stop后结束、还是错误导致的结束，都可能触发onend
    recognition.onend = () => {
      if (cancelledRef.current) {
        setStatus("idle");
        return;
      }

      // 如果是已经失败了，onerror已经处理了错误了，这里无需再处理
      if (recognitionFailedRef.current) {
        setStatus("idle");
        return;
      }

      const finalTranscript = latestTranscriptRef.current.trim();

      if (finalTranscript) {
        onTranscript(finalTranscript);
        setError("");
      } else {
        setError("没有识别到有效语音，请再试一次。");
      }

      setStatus("idle");
    };

    recognitionRef.current = recognition;

    return () => {
      cancelledRef.current = true;
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      recognition.abort();
      recognitionRef.current = null;
    };
  }, [onTranscript]);

  return {
    error,
    isSupported,
    status,
    transcript,
    toggle,
  };
};
