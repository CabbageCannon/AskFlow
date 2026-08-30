import { useChatStore } from "@/store/chatStore";
import { Virtuoso, type VirtuosoHandle } from "react-virtuoso";
import MessageRow from "./MessageRow/MessageRow";
import { useEffect, useRef } from "react";
import type { WheelEvent, TouchEvent } from "react";
import "./MessageList.css";

type MessageListProps = {
  isGenerating: boolean;
  onRegenerate: (messageId: string) => Promise<void>;
}

const MessageList = ({ isGenerating, onRegenerate }: MessageListProps) => {
  // 主动控制Virtuoso
  const virtuosoRef = useRef<VirtuosoHandle>(null)
  // 用户是否希望继续跟随
  const shouldFollowRef = useRef(true)
  // 移动端判断用户手指滚动方向
  const lastTouchYRef = useRef<number | null>(null)
  const currentSessionId = useChatStore(state => state.currentSession);
  const sessions = useChatStore(state => state.sessions);
  const currentMessages = sessions.find((session) => session.id === currentSessionId)?.messages ?? [];
  const isAtBottom = useChatStore(state => state.isAtBottom);
  const setIsAtBottom = useChatStore(state => state.setIsAtBottom);
  const lastMessage = currentMessages[currentMessages.length - 1]
  const handleWheel = (event: WheelEvent<HTMLDivElement>) => {
    // deltay小于零表示鼠标滚轮向上，用户想看历史消息，所以不应该跟随底部
    if (event.deltaY < 0) shouldFollowRef.current = false
  }
  // 处理手指滑动开始
  const handleTouchStart = (event: TouchEvent<HTMLDivElement>) => {
    lastTouchYRef.current = event.touches[0]?.clientY ?? null
  }
  // 处理手指滑动过程
  const handleTouchMove = (
    event: TouchEvent<HTMLDivElement>
  ) => {
    const currentY =
      event.touches[0]?.clientY;

    const previousY =
      lastTouchYRef.current;

    if (
      currentY !== undefined &&
      previousY !== null &&
      currentY > previousY + 2
    ) {
      shouldFollowRef.current = false;
    }

    if (currentY !== undefined) {
      lastTouchYRef.current = currentY;
    }
  };
  // 处理手指滑动结束
  const handleTouchEnd = () => {
    lastTouchYRef.current = null;
  };

  // 监听aiMessage内容是否增多
  useEffect(() => {
    if (!isGenerating || !isAtBottom) return

    // 如果正在生成内容并且当前状态正在底部，则随着item高度增加自动滚动到底部
    const frameId = requestAnimationFrame(() => {
      virtuosoRef.current?.autoscrollToBottom();
    });

    return () => {
      cancelAnimationFrame(frameId)
    }
  }, [
    lastMessage?.content,
    lastMessage?.researchProgress?.label,
    currentMessages.length
  ])

  // 监听用户是否发送了新消息，如果是的话就要直接回到底部
  useEffect(() => {
    if (!isGenerating) return

    shouldFollowRef.current = true

    const frameId = requestAnimationFrame(() => {
      virtuosoRef.current?.autoscrollToBottom()
    })

    return () => cancelAnimationFrame(frameId)
  })

  // 切换会话时也自动回到底部
  useEffect(() => {
    shouldFollowRef.current = true;

    const frameId =
      requestAnimationFrame(() => {
        virtuosoRef.current
          ?.autoscrollToBottom();
      });

    return () =>
      cancelAnimationFrame(frameId);

  }, [currentSessionId]);

  return (
  <div 
  className="message-list"
  onWheelCapture={handleWheel}

  onTouchStart={handleTouchStart}

  onTouchMoveCapture={handleTouchMove}

  onTouchEndCapture={handleTouchEnd}
  >
    <Virtuoso
      className="chat-virtuoso"
      data={currentMessages}
      ref={virtuosoRef}
      computeItemKey={(_, message) => message.id}
      itemContent={(_, message) => {
        return <MessageRow
          message={message}
          isGenerating={isGenerating}
          onRegenerate={onRegenerate}
        />
      }}
      followOutput={() => {
        return shouldFollowRef.current ? "auto" : false
      }}
      // 离底部80px左右也认为在底部
      atBottomThreshold={80}
      atBottomStateChange={(bottom) => {
        setIsAtBottom(bottom)

        if (bottom) shouldFollowRef.current = true
      }}
      overscan={50}
    />
  </div>)
}

export default MessageList;
