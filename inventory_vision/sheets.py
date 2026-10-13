"""寫入 Google 試算表：即時庫存（覆寫）+ 掃描紀錄（追加）。

斷網或 API 錯誤時，歷史紀錄會先寫入本機 JSONL 暫存檔，下次成功連線時自動補上傳，
確保紀錄不遺失；即時庫存表則每次都以最新數字覆寫，不需補傳。

每次寫入只用 3 個 API 請求（清除、覆寫、追加），多鏡頭時也是整批一次寫入，
不會超過 Google Sheets API 每分鐘的寫入配額。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

CURRENT_HEADER = ["區域", "箱數", "原始偵測數", "更新時間"]
LOG_HEADER = ["時間", "區域", "箱數", "原始偵測數"]


def _column_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


class SheetsWriter:
    def __init__(self, cfg: dict, client=None, current_header=CURRENT_HEADER, log_header=LOG_HEADER):
        self.cfg = cfg
        self.current_header = list(current_header)
        self.log_header = list(log_header)
        self.pending_path = Path(cfg["pending_file"])
        self._client = client
        self._spreadsheet = None
        self._ws_current = None
        self._ws_log = None
        self._last_logged: dict[str, int] | None = None

    # ── 連線 ────────────────────────────────────────────
    def _connect(self):
        if self._ws_current is not None:
            return
        if self._client is None:
            import gspread

            self._client = gspread.service_account(filename=self.cfg["credentials_file"])
        self._spreadsheet = self._client.open_by_key(self.cfg["spreadsheet_id"])
        self._ws_current = self._get_or_create(self.cfg["current_worksheet"], self.current_header)
        self._ws_log = self._get_or_create(self.cfg["log_worksheet"], self.log_header)

    def _get_or_create(self, title: str, header: list[str]):
        try:
            ws = self._spreadsheet.worksheet(title)
        except Exception as e:
            if type(e).__name__ != "WorksheetNotFound":
                raise
            log.info("建立工作表：%s", title)
            ws = self._spreadsheet.add_worksheet(title=title, rows=1000, cols=len(header))
            ws.append_row(header)
        return ws

    def _reset(self):
        self._ws_current = self._ws_log = self._spreadsheet = None

    # ── 暫存檔 ──────────────────────────────────────────
    def _load_pending(self) -> list[list]:
        if not self.pending_path.exists():
            return []
        rows = []
        for line in self.pending_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _save_pending(self, rows: list[list]) -> None:
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        with self.pending_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 寫入 ────────────────────────────────────────────
    def write_rows(self, current_rows: list[list], log_rows: list[list]) -> bool:
        """覆寫即時庫存表（不含標題列）並追加歷史紀錄。回傳 True 表示成功寫入雲端。"""
        log_rows = self._load_pending() + list(log_rows)
        try:
            self._connect()
            # 只清除表頭寬度的欄位再整批覆寫，區域/鏡頭增減時不會留下舊列
            self._ws_current.batch_clear([f"A:{_column_letter(len(self.current_header))}"])
            self._ws_current.update(
                values=[self.current_header] + current_rows, range_name="A1", value_input_option="USER_ENTERED"
            )
            if log_rows:
                self._ws_log.append_rows(log_rows, value_input_option="USER_ENTERED")
        except Exception as e:  # 網路、權限、配額錯誤等
            log.warning("寫入 Google 試算表失敗，%d 列紀錄暫存於本機：%s", len(log_rows), e)
            self._save_pending(log_rows)
            self._reset()
            return False

        if self.pending_path.exists():
            self.pending_path.unlink()
        return True

    def write(self, timestamp: str, counts: dict[str, int], raw_counts: dict[str, int]) -> bool:
        """單機模式：一個鏡頭、多個區域。"""
        current_rows = [[name, counts[name], raw_counts.get(name, 0), timestamp] for name in counts]
        current_rows.append(["合計", sum(counts.values()), sum(raw_counts.values()), timestamp])

        log_rows = []
        if not (self.cfg.get("log_only_on_change") and counts == self._last_logged):
            log_rows = [[timestamp, name, counts[name], raw_counts.get(name, 0)] for name in counts]
            self._last_logged = dict(counts)
        return self.write_rows(current_rows, log_rows)
