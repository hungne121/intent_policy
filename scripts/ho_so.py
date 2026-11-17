"""Sổ đăng ký văn bản của hồ sơ đề tài: tra cứu, đối chiếu, sinh mục lục.

Quy chế: docs/ho_so/QUY_CHE_HO_SO.md. Chạy từ gốc repo:
    ./run.sh scripts/ho_so.py tra --hm DL --pham-vi C4
    ./run.sh scripts/ho_so.py tra --tu 2026-11-05
    ./run.sh scripts/ho_so.py kiem
"""
import argparse
import datetime
import os
import re
import sys
import urllib.parse
from collections import defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS = os.path.join(ROOT, "docs", "ho_so")
SO = os.path.join(HS, "so_dang_ky.yaml")
PIF = os.path.join(ROOT, "docs", "PIF.md")

HM = {
    "TT": ("00_tong_the", "Tổng thể đề án"),
    "DL": ("01_thu_du_lieu", "Thu dữ liệu"),
    "DB": ("02_du_bao", "Dự báo (ManiCast)"),
    "HL": ("03_train", "Train"),
    "KT": ("04_policy_test", "Chạy policy và test"),
    "VH": ("05_van_hanh", "Vận hành"),
    "LV": ("06_luan_van", "Luận văn"),
}
LOAI = {"ĐA": "DA", "KH": "KH", "QĐ": "QD", "QC": "QC", "HD": "HD", "TTr": "TTR", "BC": "BC", "BB": "BB"}
RANG_BUOC = {"ĐA", "KH", "QĐ", "QC", "HD"}
TRANG_THAI = {
    "hieu_luc": "hiệu lực",
    "da_sua": "hiệu lực, đã sửa một phần",
    "bi_thay": "bị thay",
    "bi_bo": "bị bãi bỏ",
    "het_hieu_luc": "hết hiệu lực",
    "cho_quyet": "chờ quyết",
    "luu_tru": "lưu trữ",
}
CON_HIEU_LUC = {"hieu_luc", "da_sua"}
QUAN_HE = ("thay", "sua", "bo", "can_cu", "lien_quan")
TEN_QUAN_HE = {"thay": "thay", "sua": "sửa", "bo": "bãi bỏ", "can_cu": "căn cứ", "lien_quan": "liên quan"}
# Tài liệu không cần vào sổ: thư mục luận văn và các tệp sinh tự động của hồ sơ.
BO_QUA_TEP = re.compile(r"^docs/(thesis/|ho_so/du_thao/|ho_so/(README\.md|DOI_CHIEU\.md|[^/]+/MUC_LUC\.md)$)")
DU_THAO = os.path.join(HS, "du_thao")


def ngay_key(s):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})(?:\s+~?(\d{1,2}|\dx):(\d\d|\dx|xx))?", str(s))
    if not m:
        return None
    y, mo, d, h, mi = m.groups()
    h, mi = (h or "0").replace("x", "0"), (mi or "0").replace("x", "0")
    return (int(y), int(mo), int(d), int(h), int(mi))


def ngay_ngan(s):
    k = ngay_key(s)
    if not k:
        return str(s)
    t = re.search(r"\s(~?(?:\d{1,2}|\dx):(?:\d\d|\dx|xx))", str(s))
    return f"{k[2]:02d}/{k[1]:02d}/{k[0]}" + (f" {t.group(1)}" if t else "")


def so_hieu(v):
    m = re.match(r"^[A-Z]+-([A-Z]{2})-(\d+)$", v["so"])
    return f"{int(m.group(2)):02d}/{v['loai']}-{m.group(1)}" if m else v["so"]


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


class So:
    def __init__(self, path=SO):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.vb = data.get("van_ban") or []
        self.bo_qua = data.get("bo_qua_moc") or []
        self.by = {v["so"]: v for v in self.vb if "so" in v}
        self._lines = {}
        for v in self.vb:
            v["hm"] = as_list(v.get("hm"))
            v["moc"] = as_list(v.get("moc"))
            v["moc_pif"] = as_list(v.get("moc_pif"))
            v["pham_vi"] = as_list(v.get("pham_vi"))
            v["chu_de"] = as_list(v.get("chu_de"))
            for q in QUAN_HE:
                v[q] = as_list(v.get(q))
        self.dong = {v["so"]: self.tim_dong(v) for v in self.vb if "so" in v}

    def lines(self, rel):
        if rel not in self._lines:
            p = os.path.join(ROOT, rel)
            self._lines[rel] = open(p, encoding="utf-8").read().split("\n") if os.path.isfile(p) else None
        return self._lines[rel]

    def tim_dong(self, v):
        """Số dòng của mốc đầu tiên trong nơi lưu (None nếu không có mốc hoặc không tìm thấy)."""
        noi = v.get("noi_luu", "")
        if not v["moc"] or noi.startswith(("http", "bo_nho:")):
            return None
        L = self.lines(noi)
        if L is None:
            return None
        for i, l in enumerate(L, 1):
            if v["moc"][0] in l:
                return i
        return None

    def link(self, v, tu_thu_muc):
        noi = v.get("noi_luu", "")
        if noi.startswith("http"):
            return f"[trang]({noi})"
        if noi.startswith("bo_nho:"):
            return f"bộ nhớ agent `{noi[7:]}`"
        rel = os.path.relpath(os.path.join(ROOT, noi), tu_thu_muc)
        dong = self.dong.get(v["so"])
        url = urllib.parse.quote(rel) + (f"#L{dong}" if dong else "")
        return f"[{os.path.basename(noi)}{f':{dong}' if dong else ''}]({url})"

    def ap_dung(self, v):
        return v["loai"] in RANG_BUOC and v.get("trang_thai") in CON_HIEU_LUC


# ─────────────────────────── kiểm ───────────────────────────

def kiem(so):
    loi, canh_bao = [], []
    hom_nay = datetime.date.today()
    ids = set()
    for v in so.vb:
        s = v.get("so", "?")
        for k in ("so", "loai", "hm", "ngay", "trich_yeu", "trang_thai"):
            if not v.get(k):
                loi.append(f"`{s}` thiếu trường `{k}`")
        if s in ids:
            loi.append(f"`{s}` trùng mã")
        ids.add(s)
        if v.get("loai") not in LOAI:
            loi.append(f"`{s}` loại `{v.get('loai')}` không có trong quy chế")
        elif not re.match(rf"^{LOAI[v['loai']]}-{v['hm'][0] if v['hm'] else '??'}-\d+$", s):
            loi.append(f"`{s}` mã không khớp loại `{v['loai']}` và hạng mục chính `{v['hm'][:1]}`")
        for h in v["hm"]:
            if h not in HM:
                loi.append(f"`{s}` hạng mục `{h}` không có trong quy chế")
        if v.get("trang_thai") not in TRANG_THAI:
            loi.append(f"`{s}` trạng thái `{v.get('trang_thai')}` không hợp lệ")
        if not ngay_key(v.get("ngay", "")):
            loi.append(f"`{s}` ngày `{v.get('ngay')}` sai dạng YYYY-MM-DD [HH:MM]")
        if v.get("loai") in {"ĐA", "KH", "QĐ", "QC"} and not v.get("cap"):
            loi.append(f"`{s}` văn bản ràng buộc chưa ghi cấp")
        if v.get("loai") == "TTr" and v.get("trang_thai") not in {"cho_quyet", "het_hieu_luc"}:
            loi.append(f"`{s}` tờ trình chỉ có trạng thái cho_quyet hoặc het_hieu_luc")
        noi = v.get("noi_luu", "")
        if not noi:
            loi.append(f"`{s}` thiếu nơi lưu")
        elif not noi.startswith(("http", "bo_nho:")):
            if not os.path.exists(os.path.join(ROOT, noi)):
                loi.append(f"`{s}` nơi lưu không tồn tại: `{noi}`")
            elif v["moc"]:
                L = so.lines(noi) or []
                for m in v["moc"]:
                    if not any(m in l for l in L):
                        loi.append(f"`{s}` không thấy mốc `{m}` trong `{noi}`")
        for m in v["moc_pif"]:
            if not any(m in l for l in so.lines("docs/PIF.md") or []):
                loi.append(f"`{s}` không thấy mốc `{m}` trong `docs/PIF.md`")

    # quan hệ
    bi = defaultdict(lambda: defaultdict(list))  # y -> quan hệ -> [x]
    for x in so.vb:
        for q in QUAN_HE:
            for y in x[q]:
                if y not in so.by:
                    loi.append(f"`{x['so']}` {TEN_QUAN_HE[q]} `{y}` nhưng sổ không có mã này")
                    continue
                bi[y][q].append(x["so"])
                Y = so.by[y]
                if q != "lien_quan" and ngay_key(x["ngay"]) and ngay_key(Y["ngay"]) and ngay_key(x["ngay"]) < ngay_key(Y["ngay"]):
                    loi.append(f"`{x['so']}` ({ngay_ngan(x['ngay'])}) {TEN_QUAN_HE[q]} `{y}` ban hành sau nó ({ngay_ngan(Y['ngay'])})")
                if q == "thay" and Y["trang_thai"] != "bi_thay":
                    loi.append(f"`{x['so']}` thay `{y}` nhưng `{y}` còn trạng thái `{Y['trang_thai']}` (phải là bi_thay)")
                if q == "bo" and Y["trang_thai"] != "bi_bo":
                    loi.append(f"`{x['so']}` bãi bỏ `{y}` nhưng `{y}` còn trạng thái `{Y['trang_thai']}` (phải là bi_bo)")
                if q == "sua" and Y["trang_thai"] == "hieu_luc":
                    loi.append(f"`{x['so']}` sửa `{y}` nhưng `{y}` còn ghi `hieu_luc` (phải là da_sua)")
                if q in ("thay", "sua", "bo") and x.get("cap") and Y.get("cap") and x["cap"] > Y["cap"] \
                        and str(x.get("ban_hanh", "")).startswith("agent"):
                    canh_bao.append(f"`{x['so']}` (cấp {x['cap']}, agent ban hành) {TEN_QUAN_HE[q]} `{y}` (cấp {Y['cap']}): cần người dùng xác nhận")
                if q == "can_cu" and so.ap_dung(x) and Y["trang_thai"] in {"bi_thay", "bi_bo"}:
                    canh_bao.append(f"`{x['so']}` còn hiệu lực nhưng căn cứ `{y}` đã {TRANG_THAI[Y['trang_thai']]}")
    for v in so.vb:
        t = v.get("trang_thai")
        if t == "bi_thay" and not bi[v["so"]]["thay"]:
            loi.append(f"`{v['so']}` ghi bi_thay nhưng không văn bản nào ghi thay nó")
        if t == "bi_bo" and not bi[v["so"]]["bo"]:
            loi.append(f"`{v['so']}` ghi bi_bo nhưng không văn bản nào ghi bãi bỏ nó")
        if t == "da_sua" and not bi[v["so"]]["sua"]:
            loi.append(f"`{v['so']}` ghi da_sua nhưng không văn bản nào ghi sửa nó")

    # cùng chủ đề: văn bản còn hiệu lực phải nối với nhau bằng quan hệ
    nhom = defaultdict(list)
    for v in so.vb:
        if so.ap_dung(v):
            for c in v["chu_de"]:
                nhom[c].append(v)
    chu_de_ho = []
    for c, vs in sorted(nhom.items()):
        if len(vs) < 2:
            continue
        ma = {v["so"] for v in vs}
        ke = {m: set() for m in ma}
        for v in vs:
            for q in QUAN_HE:
                for y in v[q]:
                    if y in ma:
                        ke[v["so"]].add(y)
                        ke[y].add(v["so"])
        thanh_phan, da = [], set()
        for m in sorted(ma):
            if m in da:
                continue
            st, tp = [m], set()
            while st:
                a = st.pop()
                if a in tp:
                    continue
                tp.add(a)
                st.extend(ke[a] - tp)
            da |= tp
            thanh_phan.append(sorted(tp))
        if len(thanh_phan) > 1:
            thu_tu = sorted(vs, key=lambda v: (v.get("cap") or 9, tuple(-x for x in ngay_key(v["ngay"]))))
            chu_de_ho.append((c, thanh_phan, thu_tu))

    # văn bản chờ quyết
    cho = sorted((v for v in so.vb if v.get("trang_thai") == "cho_quyet"), key=lambda v: ngay_key(v["ngay"]))
    tuoi = {v["so"]: (hom_nay - datetime.date(*ngay_key(v["ngay"])[:3])).days for v in cho}

    # quyết định trong PIF chưa vào sổ
    moc_all = [m for v in so.vb for m in v["moc"] + v["moc_pif"]] + [b["moc"] for b in so.bo_qua]
    chua_so = []
    for i, l in enumerate(so.lines("docs/PIF.md") or [], 1):
        s = l.strip()
        la_qd = (s.startswith(("|", "**", "- **")) and "người dùng" in s.lower()) or "**Quyết định" in s
        if la_qd and not any(m in l for m in moc_all):
            chua_so.append((i, s[:150]))

    # tài liệu chưa vào sổ
    tham_chieu = " ".join(f"{v.get('noi_luu', '')} {v.get('ghi_chu', '')}" for v in so.vb)
    tai_lieu_chua = []
    for d, _, fs in os.walk(os.path.join(ROOT, "docs")):
        for f in fs:
            if not f.endswith(".md"):
                continue
            rel = os.path.relpath(os.path.join(d, f), ROOT)
            if BO_QUA_TEP.match(rel):
                continue
            duoi = "/".join(rel.split("/")[-2:])
            if rel not in tham_chieu and duoi not in tham_chieu and f not in tham_chieu:
                tai_lieu_chua.append(rel)

    bo_nho = [v for v in so.vb if v.get("noi_luu", "").startswith("bo_nho:") and so.ap_dung(v)]
    du_thao = sorted(f for f in os.listdir(DU_THAO) if f.endswith(".md")) if os.path.isdir(DU_THAO) else []
    return dict(loi=loi, canh_bao=canh_bao, chu_de_ho=chu_de_ho, cho=cho, tuoi=tuoi,
                chua_so=chua_so, tai_lieu_chua=sorted(tai_lieu_chua), bo_nho=bo_nho, du_thao=du_thao)


# ─────────────────────────── sinh tài liệu ───────────────────────────

def dong_bang(so, v, tu):
    ty = v["trich_yeu"].replace("|", "/")
    lq = [f"{TEN_QUAN_HE[q]} {', '.join(f'`{so_hieu(so.by[y])}`' if y in so.by else y for y in v[q])}"
          for q in ("thay", "sua", "bo") if v[q]]
    if lq:
        ty += f" *({'; '.join(lq)})*"
    return f"| {so_hieu(v)} | {ngay_ngan(v['ngay'])} | {ty} | {TRANG_THAI[v['trang_thai']]} | {so.link(v, tu)} |"


DAU_BANG = "| Số hiệu | Ngày | Trích yếu | Trạng thái | Toàn văn |\n|---|---|---|---|---|"


def sap(vs):
    return sorted(vs, key=lambda v: tuple(-x for x in ngay_key(v["ngay"])))


def muc_luc(so, kq, gio):
    for ma, (thu_muc, ten) in HM.items():
        tu = os.path.join(HS, thu_muc)
        os.makedirs(tu, exist_ok=True)
        vs = [v for v in so.vb if ma in v["hm"]]
        out = [f"# {ten} — mục lục ({ma})", "",
               f"Sinh tự động từ `docs/ho_so/so_dang_ky.yaml` lúc {gio}. Không sửa tay; sửa sổ rồi chạy `./run.sh scripts/ho_so.py kiem`.",
               f"Tra nhanh: `./run.sh scripts/ho_so.py tra --hm {ma}` (thêm `--pham-vi <task>` để lọc theo task).", ""]
        ap = [v for v in vs if so.ap_dung(v)]
        out.append("## Đang hiệu lực")
        out.append("")
        for cap in (1, 2, 3):
            nhom = sap([v for v in ap if v.get("cap") == cap])
            if nhom:
                out += [f"### Cấp {cap}", "", DAU_BANG] + [dong_bang(so, v, tu) for v in nhom] + [""]
        cho = sap([v for v in vs if v.get("trang_thai") == "cho_quyet"])
        if cho:
            out += ["## Chờ quyết (tờ trình, biên bản chờ ký)", "", DAU_BANG] + [dong_bang(so, v, tu) for v in cho] + [""]
        bb = sap([v for v in vs if v["loai"] == "BB" and v.get("trang_thai") != "cho_quyet"])
        if bb:
            out += ["## Biên bản nghiệm thu", "", DAU_BANG] + [dong_bang(so, v, tu) for v in bb] + [""]
        bc = sap([v for v in vs if v["loai"] == "BC" and v.get("trang_thai") == "hieu_luc"])
        if bc:
            out += ["## Báo cáo hiện hành", "", DAU_BANG] + [dong_bang(so, v, tu) for v in bc] + [""]
        cu = sap([v for v in vs if v.get("trang_thai") in {"bi_thay", "bi_bo", "het_hieu_luc", "luu_tru"}])
        if cu:
            out += ["<details><summary>Hết hiệu lực, bị thay, lưu trữ (" + str(len(cu)) + ")</summary>", "",
                    DAU_BANG] + [dong_bang(so, v, tu) for v in cu] + ["", "</details>", ""]
        open(os.path.join(tu, "MUC_LUC.md"), "w", encoding="utf-8").write("\n".join(out))

    out = ["# Hồ sơ đề tài ACT + ManiCast", "",
           f"Sinh tự động lúc {gio} từ [sổ đăng ký](so_dang_ky.yaml). Quy định cách quản lý: [quy chế](QUY_CHE_HO_SO.md). "
           f"Kết quả đối chiếu gần nhất: [DOI_CHIEU.md](DOI_CHIEU.md).", "",
           "## Tra theo việc đang làm", "",
           "| Hạng mục | Mục lục | Đang hiệu lực | Chờ quyết |", "|---|---|---|---|"]
    for ma, (thu_muc, ten) in HM.items():
        vs = [v for v in so.vb if ma in v["hm"]]
        out.append(f"| {ma} — {ten} | [{thu_muc}/MUC_LUC.md]({thu_muc}/MUC_LUC.md) | "
                   f"{sum(so.ap_dung(v) for v in vs)} | {sum(v.get('trang_thai') == 'cho_quyet' for v in vs)} |")
    if kq["du_thao"]:
        out += ["", "## Dự thảo chờ duyệt (chưa vào sổ)", ""] + [f"- [{f}](du_thao/{urllib.parse.quote(f)})" for f in kq["du_thao"]]
    out += ["", "## Chờ người dùng quyết", "", DAU_BANG] + [dong_bang(so, v, HS) for v in sap(kq["cho"])]
    out += ["", "## Mới nhất (10 văn bản)", "", DAU_BANG] + [dong_bang(so, v, HS) for v in sap(so.vb)[:10]] + [""]
    open(os.path.join(HS, "README.md"), "w", encoding="utf-8").write("\n".join(out))


def ghi_doi_chieu(so, kq, gio):
    out = [f"# Đối chiếu hồ sơ — {gio}", "",
           "Sinh tự động bởi `./run.sh scripts/ho_so.py kiem`. Mục A, B cần người dùng xem; mục C–F là việc của agent.", "",
           f"Sổ có {len(so.vb)} văn bản; {sum(so.ap_dung(v) for v in so.vb)} đang hiệu lực; {len(kq['cho'])} chờ quyết.", ""]
    out += ["## A. Chờ người dùng quyết", ""]
    out += [f"- **{so_hieu(v)}** ({ngay_ngan(v['ngay'])}, {kq['tuoi'][v['so']]} ngày): {v['trich_yeu']}" for v in kq["cho"]] or ["- Không có."]
    out += ["", "## B. Cùng chủ đề còn hiệu lực nhưng chưa liên kết (có thể xung đột)", "",
            "Thứ tự áp dụng theo quy chế §6: cấp cao trước, cùng cấp thì văn bản sau trước.", ""]
    if kq["chu_de_ho"]:
        for c, tp, thu_tu in kq["chu_de_ho"]:
            out.append(f"- `{c}`: " + " · ".join(f"{so_hieu(v)} (cấp {v.get('cap', '?')}, {ngay_ngan(v['ngay'])})" for v in thu_tu)
                       + f" — {len(tp)} nhóm chưa nối: " + " / ".join(", ".join(so_hieu(so.by[m]) for m in g) for g in tp))
    else:
        out.append("- Không có.")
    out += ["", "## C. Lỗi sổ", ""] + ([f"- {x}" for x in kq["loi"]] or ["- Không có."])
    out += ["", "## D. Cảnh báo", ""] + ([f"- {x}" for x in kq["canh_bao"]] or ["- Không có."])
    out += ["", "## E. Dòng có dạng quyết định trong PIF chưa vào sổ", "",
            "Đăng ký vào sổ, hoặc ghi vào `bo_qua_moc` kèm lý do nếu không phải văn bản.", ""]
    out += [f"- PIF dòng {i}: {s}" for i, s in kq["chua_so"]] or ["- Không có."]
    out += ["", "## F. Tài liệu trong docs/ chưa vào sổ", ""] + ([f"- `{x}`" for x in kq["tai_lieu_chua"]] or ["- Không có."])
    out += ["", "## G. Quyết định còn hiệu lực chỉ có trong bộ nhớ agent (chưa có toàn văn trong repo)", ""]
    out += [f"- {so_hieu(v)}: {v['trich_yeu']} (`{v['noi_luu'][7:]}`)" for v in kq["bo_nho"]] or ["- Không có."]
    out += ["", "## H. Dự thảo chờ người dùng duyệt (chưa vào sổ)", ""]
    out += [f"- [{f}](du_thao/{urllib.parse.quote(f)})" for f in kq["du_thao"]] or ["- Không có."]
    open(os.path.join(HS, "DOI_CHIEU.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")


# ─────────────────────────── tra cứu ───────────────────────────

def tra(so, a):
    vs = so.vb
    if a.hm:
        vs = [v for v in vs if a.hm in v["hm"]]
    if a.pham_vi:
        p = a.pham_vi
        vs = [v for v in vs if not v["pham_vi"] or "tat_ca" in v["pham_vi"] or p in v["pham_vi"]
              or f"task.{p}" in v["chu_de"]]
    if a.chu_de:
        vs = [v for v in vs if any(c.startswith(a.chu_de) for c in v["chu_de"])]
    if a.tu:
        k = ngay_key(a.tu)
        vs = sap([v for v in vs if ngay_key(v["ngay"]) >= k])
        tieu_de = f"Văn bản ban hành từ {a.tu} (mới trước)"
    else:
        cho = [v for v in vs if v.get("trang_thai") == "cho_quyet"]
        if not a.tat_ca:
            vs = [v for v in vs if so.ap_dung(v)]
        vs = sorted(vs, key=lambda v: (v.get("cap") or 9, tuple(-x for x in ngay_key(v["ngay"]))))
        tieu_de = "Văn bản " + ("(mọi trạng thái)" if a.tat_ca else "đang hiệu lực") + ", cấp cao trước, cùng cấp thì mới trước"
    loc = ", ".join(x for x in (a.hm and f"hạng mục {a.hm}", a.pham_vi and f"phạm vi {a.pham_vi}",
                                a.chu_de and f"chủ đề {a.chu_de}") if x)
    print(f"{tieu_de}{' — ' + loc if loc else ''}: {len(vs)}\n")
    for v in vs:
        dong = so.dong.get(v["so"])
        noi = v.get("noi_luu", "") + (f":{dong}" if dong else "")
        cap = f"cấp {v['cap']}" if v.get("cap") else v["loai"]
        print(f"{so_hieu(v):<11} {ngay_ngan(v['ngay']):<17} {cap:<6} {TRANG_THAI[v['trang_thai']]}")
        print(f"    {v['trich_yeu']}")
        lq = [f"{TEN_QUAN_HE[q]} {', '.join(so_hieu(so.by[y]) for y in v[q] if y in so.by)}" for q in ("thay", "sua", "bo") if v[q]]
        print(f"    → {noi}" + (f"   ({'; '.join(lq)})" if lq else ""))
    if not a.tu and cho:
        print(f"\nChờ quyết trong phạm vi này: " + ", ".join(so_hieu(v) for v in sap(cho)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="lenh", required=True)
    t = sub.add_parser("tra", help="tra văn bản đang hiệu lực / mới ban hành")
    t.add_argument("--hm", choices=sorted(HM), help="hạng mục")
    t.add_argument("--pham-vi", help="task / dataset / mô hình, ví dụ C4, v10b, D")
    t.add_argument("--chu-de", help="tiền tố chủ đề, ví dụ cong_va_cham, task.C1")
    t.add_argument("--tu", help="chỉ văn bản ban hành từ ngày YYYY-MM-DD (mọi trạng thái)")
    t.add_argument("--tat-ca", action="store_true", help="kể cả văn bản hết hiệu lực")
    sub.add_parser("kiem", help="đối chiếu sổ, ghi DOI_CHIEU.md và mục lục")
    a = ap.parse_args()
    so = So()
    if a.lenh == "tra":
        tra(so, a)
        return
    gio = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    kq = kiem(so)
    muc_luc(so, kq, gio)
    ghi_doi_chieu(so, kq, gio)
    print(f"Đối chiếu {gio}: {len(so.vb)} văn bản, {sum(so.ap_dung(v) for v in so.vb)} hiệu lực, {len(kq['cho'])} chờ quyết")
    print(f"  lỗi sổ {len(kq['loi'])}, cảnh báo {len(kq['canh_bao'])}, chủ đề chưa liên kết {len(kq['chu_de_ho'])}, "
          f"dòng PIF chưa vào sổ {len(kq['chua_so'])}, tài liệu chưa vào sổ {len(kq['tai_lieu_chua'])}, "
          f"chỉ trong bộ nhớ {len(kq['bo_nho'])}, dự thảo chờ duyệt {len(kq['du_thao'])}")
    print("  → docs/ho_so/DOI_CHIEU.md, docs/ho_so/README.md, docs/ho_so/*/MUC_LUC.md")
    sys.exit(1 if kq["loi"] else 0)


if __name__ == "__main__":
    main()
