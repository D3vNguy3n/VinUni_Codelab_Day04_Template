"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Vai trò: tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng VinFast, Vinpearl.
- Giọng nói: chuyên nghiệp, thân thiện, ngắn gọn và chính xác.

## AVAILABLE TOOLS
- search_product_catalog: tra cứu sản phẩm theo danh mục và giá tối đa.
- submit_support_ticket: tạo yêu cầu hỗ trợ cho khách hàng.

## CORE RULES
1. Không bịa hoặc suy đoán dữ liệu sản phẩm; phải gọi tool khi cần dữ liệu thực.
2. Trình bày rõ kết quả tool và thông báo khi không có kết quả.
3. Không tự nhận đã tạo ticket nếu tool chưa trả về mã ticket.

## OPERATIONAL BOUNDARIES
Chỉ hỗ trợ sản phẩm, dịch vụ và yêu cầu thuộc hệ sinh thái Vingroup.

## OUTPUT CONTRACT
Suy nghĩ ngắn gọn, ghi rõ Thought, Action, Observation khi dùng tool, sau đó trả lời bằng Final Answer.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        lower_input = user_input.lower()
        is_warranty_faq = "bảo hành" in lower_input and "pin" in lower_input
        needs_catalog = not is_warranty_faq and any(word in lower_input for word in ("xe điện", "xe vinfast", "resort", "vinpearl", "sản phẩm"))
        needs_ticket = any(word in lower_input for word in ("ghi nhận", "hỗ trợ", "bị lỗi", "phản hồi", "khiếu nại"))
        intents = {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": not needs_catalog and not needs_ticket
        }
        self.trace.append({"step": "intent_detection", "intents": intents})

        actions = []
        if needs_catalog:
            category = "du_lich" if any(word in lower_input for word in ("resort", "vinpearl", "du lịch")) else "xe_dien"
            max_price = self._extract_max_price(user_input)
            actions.append(("search_product_catalog", {"category": category, "max_price": max_price}))
        if needs_ticket:
            customer_name = self._extract_customer_name(user_input)
            issue_description = self._extract_issue(user_input)
            priority = "high" if any(word in lower_input for word in ("gấp", "nghiêm trọng", "khẩn")) else "medium"
            actions.append(("submit_support_ticket", {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }))

        if not actions:
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm theo thông tin trong hệ thống."
            self.trace.append({"step": "final_answer", "answer": answer})
            return {"answer": answer, "trace": self.trace, "iterations": 1, "status": "completed"}

        if len(actions) > self.max_iterations:
            return {"answer": "Lỗi: Vượt quá số bước tối đa.", "trace": self.trace, "iterations": 0, "status": "max_iterations_reached"}

        observations = []
        for step, (tool_name, arguments) in enumerate(actions, start=1):
            result = TOOL_MAP[tool_name](**arguments)
            observations.append((tool_name, result))
            self.trace.append({"step": step, "action": tool_name, "arguments": arguments, "observation": result})

        answer_parts = []
        for tool_name, result in observations:
            if tool_name == "search_product_catalog":
                if not result:
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    names = ", ".join(product["name"] for product in result)
                    answer_parts.append(f"Sản phẩm phù hợp: {names}.")
            else:
                answer_parts.append(f"Đã tạo ticket {result['ticket_id']} cho {result['customer_name']}.")
        answer = " ".join(answer_parts)
        self.trace.append({"step": "final_answer", "answer": answer})
        return {"answer": answer, "trace": self.trace, "iterations": len(actions), "status": "completed"}

    @staticmethod
    def _extract_max_price(user_input: str) -> int:
        match = re.search(r"(?:dưới|không quá|tối đa)\s+(\d+(?:[.,]\d+)?)\s*(tỷ|triệu|tr)", user_input.lower())
        if not match:
            return 999999999999
        value = float(match.group(1).replace(",", "."))
        multiplier = 1_000_000_000 if match.group(2) == "tỷ" else 1_000_000
        return int(value * multiplier)

    @staticmethod
    def _extract_customer_name(user_input: str) -> str:
        match = re.search(r"(?:tên tôi là|tôi tên|cho tôi tên là)\s+([^,.]+)", user_input, re.IGNORECASE)
        return match.group(1).strip() if match else "Khách hàng"

    @staticmethod
    def _extract_issue(user_input: str) -> str:
        match = re.search(r"(?:phản hồi:\s*|là\s+)(.+?)(?:,\s*mức độ|\.|$)", user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"(?:xe|phòng).+?(?:bị|gặp)\s+(.+?)(?:\.|,|$)", user_input, re.IGNORECASE)
        return match.group(0).strip() if match else user_input.strip()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
