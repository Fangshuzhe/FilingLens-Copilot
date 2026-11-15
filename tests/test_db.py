from filinglens.db import Database
from filinglens.models import Evidence, PageText, TextChunk


def test_database_round_trip(tmp_path):
    database = Database(tmp_path / "test.db")
    report_id = database.add_report(
        company="示例公司",
        year=2025,
        filename="sample.pdf",
        file_path=tmp_path / "sample.pdf",
        pages=[PageText(page_number=1, text="公司的主要业务是软件服务。")],
        chunks=[
            TextChunk(
                chunk_index=0,
                page_number=1,
                text="公司的主要业务是软件服务。",
            )
        ],
    )

    report = database.get_report(report_id)
    assert report is not None
    assert report.company == "示例公司"
    assert report.page_count == 1
    assert len(database.list_chunks(report_id)) == 1

    evidence = Evidence(
        evidence_id="E1",
        chunk_id=database.list_chunks(report_id)[0].id,
        report_id=report_id,
        page_number=1,
        text="公司的主要业务是软件服务。",
        score=1.5,
    )
    insight_id = database.add_insight(
        report_id=report_id,
        text="公司通过软件服务获得收入。",
        evidence=[evidence],
        kind="annual_report_fact",
    )

    insights = database.list_insights(report_id)
    assert insights[0].id == insight_id
    assert insights[0].evidence[0].text == evidence.text


def test_chat_messages_persist_by_conversation(tmp_path):
    database = Database(tmp_path / "chat.db")
    conversation_id = "a" * 32

    database.add_chat_message(
        conversation_id=conversation_id,
        role="user",
        content="你好",
    )
    database.add_chat_message(
        conversation_id=conversation_id,
        role="assistant",
        content="你好，我是 FilingLens。",
    )

    messages = database.list_chat_messages(conversation_id)

    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "你好"
    assert messages[1].content == "你好，我是 FilingLens。"


def test_list_conversations_keeps_old_chats_and_latest_first(tmp_path):
    database = Database(tmp_path / "chat.db")
    first_id = "a" * 32
    second_id = "b" * 32

    database.add_chat_message(
        conversation_id=first_id,
        role="user",
        content="第一段对话",
    )
    database.add_chat_message(
        conversation_id=first_id,
        role="assistant",
        content="第一段回答",
    )
    database.add_chat_message(
        conversation_id=second_id,
        role="user",
        content="第二段对话",
    )

    conversations = database.list_conversations()

    assert [item.conversation_id for item in conversations] == [
        second_id,
        first_id,
    ]
    assert conversations[0].title == "第二段对话"
    assert conversations[1].title == "第一段对话"
    assert conversations[1].message_count == 2


def test_delete_conversation_removes_only_selected_chat(tmp_path):
    database = Database(tmp_path / "chat.db")
    first_id = "a" * 32
    second_id = "b" * 32
    database.add_chat_message(
        conversation_id=first_id,
        role="user",
        content="保留",
    )
    database.add_chat_message(
        conversation_id=second_id,
        role="user",
        content="删除",
    )

    deleted = database.delete_conversation(second_id)

    assert deleted == 1
    assert database.list_chat_messages(second_id) == []
    assert len(database.list_chat_messages(first_id)) == 1

