from voice2txt.core.engine import Utterance
from voice2txt.io.formatter import format_srt, format_txt
from voice2txt.io.cleanup import clean_text, clean_utterances


def test_srt_basic():
    utts = [
        Utterance(0.0, 2.5, "你好世界", "SPEAKER_00"),
        Utterance(3.0, 5.0, "Hello world", "SPEAKER_01"),
    ]
    out = format_srt(utts, speaker_labels=True)
    assert "00:00:00,000 --> 00:00:02,500" in out
    assert "[SPEAKER_00] 你好世界" in out
    assert "[SPEAKER_01] Hello world" in out


def test_srt_no_labels():
    utts = [Utterance(0.0, 2.5, "你好世界", "SPEAKER_00")]
    out = format_srt(utts, speaker_labels=False)
    assert "[SPEAKER_00]" not in out
    assert "你好世界" in out


def test_txt_with_ts():
    utts = [Utterance(0.0, 2.5, "你好", "SPEAKER_00")]
    out = format_txt(utts, speaker_labels=True, with_timestamps=True)
    assert out.strip() == "[00:00:00,000] [SPEAKER_00] 你好"


# ---- 标点清洗 ----
def test_clean_text_strips_leading_punct():
    assert clean_text("，你好世界") == "你好世界"
    assert clean_text("。 嗯，") == "嗯"
    assert clean_text("，，，，开始") == "开始"


def test_clean_text_collapses_repeated_punct():
    assert clean_text("开饭时间早上9点至下午5点。。") == "开饭时间早上9点至下午5点。"
    assert clean_text("你好，，，我是") == "你好，我是"
    assert clean_text("真的吗？？？") == "真的吗？"


def test_clean_text_drops_pure_punct():
    assert clean_utterances([Utterance(0.0, 1.0, "，，，", "SPEAKER_00")],
                            merge_speakers=False) == []
    assert clean_utterances([Utterance(0.0, 1.0, "。", "SPEAKER_00")],
                            merge_speakers=False) == []


def test_clean_trailing_redundant_punct():
    # 块尾 "，。" -> "。"；"。，。" -> "。"；"，" -> ""
    assert clean_text("他说。，") == "他说。"
    assert clean_text("取不了机票。 ，。") == "取不了机票。"
    assert clean_text("嗯，") == "嗯"
    assert clean_text("好的。") == "好的。"


def test_clean_squeezes_space_around_punct():
    assert clean_text("机票。 ，。") == "机票。"
    assert clean_text("你好 ， 世界 。") == "你好，世界。"


# ---- TXT 合并连续同说话人 ----
def test_txt_merges_same_speaker():
    utts = [
        Utterance(0.0, 2.0, "开饭时间早上9点至下午5点。", "SPEAKER_00"),
        Utterance(2.5, 4.0, "嗯，", "SPEAKER_00"),
        Utterance(4.5, 6.0, "好的，", "SPEAKER_00"),
    ]
    out = format_txt(utts, speaker_labels=True)
    # 同一说话人合为一行，前缀只出现一次
    assert out.count("[SPEAKER_00]") == 1
    assert "开饭时间早上9点至下午5点。嗯，好的" in out


def test_txt_keeps_different_speakers_separate():
    utts = [
        Utterance(0.0, 2.0, "你好", "SPEAKER_00"),
        Utterance(2.5, 4.0, "你好", "SPEAKER_01"),
        Utterance(4.5, 6.0, "再见", "SPEAKER_00"),
    ]
    out = format_txt(utts, speaker_labels=True)
    assert out.count("[SPEAKER_00]") == 2
    assert out.count("[SPEAKER_01]") == 1


def test_txt_no_diarize_single_block():
    # 无说话人分离时所有碎片合并为一段
    utts = [
        Utterance(0.0, 2.0, "第一句。", None),
        Utterance(2.5, 4.0, "第二句。", None),
    ]
    out = format_txt(utts, speaker_labels=True)
    assert "[" not in out  # 无标签
    assert "第一句。第二句。" in out
