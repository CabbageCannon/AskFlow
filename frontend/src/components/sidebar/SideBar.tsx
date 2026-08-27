import { useState } from "react"
import icons from "@/assets/icons/icons";
import { useChatStore } from "@/store/chatStore";
import HistorySession from "../HistorySession/HistorySession";
import "./SideBar.css";

const SideBar = () => {
  const [extend, setExtend] = useState<boolean>(true);
  const createSession = useChatStore(state => state.createSession);

  const changeExtend = () => {
    setExtend(!extend);
  }

  const handleAddNewSession = () => {
    createSession();
  }

  return (
    <div className="sideBar">
      <div className="top">
        {/* 标题 */}
        <div className="title">
          {extend && <div className="titleText">ZHAI AI</div>}
          <img src={icons.menuIcon} alt="" className="icon" onClick={changeExtend} />
        </div>
        <div className="menu">
          <button type="button" className="menuItem">
            <img src={icons.addSession} alt="" className="icon" />
            {extend && <div className="addSessionText" onClick={handleAddNewSession}>添加对话</div>}
          </button>
        </div>
        {extend &&
          <HistorySession />
        }
      </div>
      <div className="bottom">
        <div className="userCard">
          <div className="touxiang">
            <img src={icons.touxiangIcon} alt="" />
          </div>
          {extend &&
            <div className="username">
              翟靖朗
            </div>}
        </div>
      </div>
    </div>
  )
}

export default SideBar
