"""M1 'Hello board': the Data Checker receives a test question and answers it."""
from shared.messages import Message, MessageType
from shared.tables import Tables
from shared.testing import InMemoryBoard
from agents.checker.agent import CheckerAgent
from agents.checker.settings import CheckerSettings


def test_checker_answers_query_data_changed(tmp_path):
    board = InMemoryBoard()
    agent = CheckerAgent(board=board, tables=Tables(":memory:"),
                         settings=CheckerSettings(data_dir=tmp_path))
    question = Message(type=MessageType.QUESTION, kind="query.data_changed", sender="monitor",
                       to=["checker"], topic="feature:total_sulfur_dioxide", level=1,
                       summary="did the data change?",
                       payload={"feature": "total_sulfur_dioxide", "start": "", "end": ""})
    board.post(question)
    agent.step()

    answers = [m for m in board.messages if m.reply_to == question.id]
    assert len(answers) == 1
    answer = answers[0]
    assert answer.type == MessageType.ANSWER and answer.sender == "checker"
    assert answer.to == ["monitor"] and answer.kind == "query.data_changed"
    assert set(answer.payload) == {"changed", "batch_ids", "likely_cause"}
