# plugin/multi_task_rewards.py
import json
import re
from swift.rewards import ORM, orms


# ============================================================
# 第一部分：公共工具函数（所有任务共用）
# ============================================================

def _extract_json(text):
    """从模型输出中鲁棒提取 JSON。
    支持：裸 JSON / ```json 代码块 / 夹杂文字中的 JSON片段。
    """
    if isinstance(text, dict):
        return text
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None

    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except Exception:
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            return None


def _is_pure_json(text: str) -> bool:
    """判断输出是否为纯 JSON（裸 JSON 或完整 markdown 代码块，不允许前后夹杂解释文字）。"""
    if not isinstance(text, str):
        return False
    s = text.strip()
    try:
        json.loads(s)
        return True
    except Exception:
        pass
    m = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.DOTALL)
    if m:
        try:
            json.loads(m.group(1).strip())
            return True
        except Exception:
            pass
    return False


def _to_bool(v):
    """将 has_anomaly 字段归一化为 Python bool，容错常见的字符串形式。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "是", "有异常", "1"):
            return True
        if s in ("false", "否", "正常", "无异常", "0"):
            return False
    return None


def _valid_bbox(obj):
    """检查 object 中的 bbox_2d 是否合法（千分位坐标）。"""
    if not isinstance(obj, dict):
        return False
    bbox = obj.get("bbox_2d")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def _iou(box1, box2):
    """计算两个 [x1, y1, x2, y2] 框的 IoU。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _parse_objects(obj_list):
    """解析 objects 列表，过滤非法框（含千分位坐标越界、x1>=x2 等）。"""
    result = []
    if not isinstance(obj_list, list):
        return result
    for o in obj_list:
        if not _valid_bbox(o):   # 复用已有校验
            continue
        bbox = o.get("bbox_2d")
        label = o.get("label", "")
        try:
            result.append(([float(x) for x in bbox], str(label)))
        except Exception:
            continue
    return result


def _detection_f1(pred_objs, gt_objs, iou_thr=0.5, use_label=False):
    """贪心 IoU 匹配，返回 F1 ∈ [0, 1]。双方皆空视为满分。"""
    preds = _parse_objects(pred_objs)
    gts = _parse_objects(gt_objs)

    if len(preds) == 0 and len(gts) == 0:
        return 1.0
    if len(preds) == 0 or len(gts) == 0:
        return 0.0

    pairs = []
    for pi, (pbox, plabel) in enumerate(preds):
        for gi, (gbox, glabel) in enumerate(gts):
            if use_label and plabel.strip() != glabel.strip():
                continue
            iou = _iou(pbox, gbox)
            if iou >= iou_thr:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)

    matched_p, matched_g = set(), set()
    tp = 0
    for _, pi, gi in pairs:
        if pi in matched_p or gi in matched_g:
            continue
        matched_p.add(pi)
        matched_g.add(gi)
        tp += 1

    precision = tp / len(preds)
    recall = tp / len(gts)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _detection_f1_and_iou(pred_objs, gt_objs, iou_thr=0.5, use_label=False):
    """返回 (F1, mean_matched_iou)。
    - F1: 带阈值，衡量检出质量
    - mean_iou: 无阈值贪心匹配，提供连续定位信号
    """
    preds = _parse_objects(pred_objs)
    gts = _parse_objects(gt_objs)

    if len(preds) == 0 and len(gts) == 0:
        return 1.0, 1.0
    if len(preds) == 0 or len(gts) == 0:
        return 0.0, 0.0

    all_pairs = []
    for pi, (pbox, plabel) in enumerate(preds):
        for gi, (gbox, glabel) in enumerate(gts):
            if use_label and plabel.strip() != glabel.strip():
                continue
            all_pairs.append((_iou(pbox, gbox), pi, gi))
    all_pairs.sort(reverse=True)

    # mean_iou：无阈值贪心匹配
    matched_p, matched_g = set(), set()
    ious = []
    for iou, pi, gi in all_pairs:
        if pi not in matched_p and gi not in matched_g:
            matched_p.add(pi)
            matched_g.add(gi)
            ious.append(iou)
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    # F1：带阈值
    matched_p, matched_g = set(), set()
    tp = 0
    for iou, pi, gi in all_pairs:
        if iou >= iou_thr and pi not in matched_p and gi not in matched_g:
            matched_p.add(pi)
            matched_g.add(gi)
            tp += 1

    precision = tp / len(preds)
    recall = tp / len(gts)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return f1, mean_iou


def _text_group_score(pred, field_names, min_len=8):
    """多个描述字段整体质量分：防退化（复读机/过短）。返回 [0, 1]。"""
    scores = []
    texts = []

    for f in field_names:
        val = pred.get(f, "")
        if not isinstance(val, str):
            scores.append(0.0)
            texts.append("")
            continue
        s = val.strip()
        texts.append(s)
        if not s:
            scores.append(0.0)
        elif len(s) < 4:
            scores.append(0.1)
        elif len(s) < min_len:
            scores.append(0.3)
        elif len(s) < 15:
            scores.append(0.7)
        else:
            unique_ratio = len(set(s)) / max(len(s), 1)
            scores.append(0.5 if unique_ratio < 0.2 else 1.0)

    if not scores:
        return 0.0

    base = sum(scores) / len(scores)

    # 多字段内容完全相同：退化惩罚
    non_empty = [t for t in texts if t]
    if len(non_empty) >= 2:
        unique_cnt = len(set(non_empty))
        if unique_cnt == 1:
            base *= 0.6
        elif unique_cnt <= len(non_empty) // 2:
            base *= 0.85

    return max(0.0, min(1.0, base))


# ============================================================
# 第二部分：各任务奖励函数
# ============================================================

class DeviceDiffReward(ORM):
    REQUIRED_FIELDS = [
        "image1_description", "image2_description", "difference_analysis",
        "has_anomaly", "anomaly_type", "reason", "objects",
    ]
    DESC_FIELDS = ["image1_description", "image2_description", "difference_analysis", "reason"]

    W_FORMAT = 0.08
    W_TEXT   = 0.07

    # 正样本（gt 有异常）：漏检重罚
    W_BINARY_POS = 0.50
    W_TYPE_POS = 0.10
    W_BBOX_POS = 0.25

    # 负样本（gt 正常）：误报轻罚；检测设备位置是核心任务
    W_BINARY_NEG = 0.25
    W_BBOX_NEG = 0.60

    def __call__(self, completions, task=None, solution=None, **kwargs):
        tasks = task if task is not None else kwargs.get("task", [None] * len(completions))
        solutions = solution if solution is not None else kwargs.get("solution", [None] * len(completions))
        rewards = []
        for completion, t, sol in zip(completions, tasks, solutions):
            if not (isinstance(t, str) and t.startswith("diff_")):
                rewards.append(None)
                continue
            rewards.append(self._score(completion, sol))
        return rewards

    def _score(self, completion, solution):
        pred = _extract_json(completion)
        gt = _extract_json(solution) if isinstance(solution, str) else solution
        if pred is None or gt is None:
            return 0.0

        reward = 0.0

        # 1. 格式分
        fields_ok = all(f in pred for f in self.REQUIRED_FIELDS)
        bool_ok = _to_bool(pred.get("has_anomaly")) is not None
        objects_ok = isinstance(pred.get("objects"), list)
        if fields_ok and bool_ok and objects_ok:
            reward += self.W_FORMAT
        elif bool_ok:
            reward += self.W_FORMAT * 0.5

        # 2. 描述质量（保持项）
        reward += self.W_TEXT * _text_group_score(pred, self.DESC_FIELDS)

        # 3. 核心：类别条件评分
        pred_flag = _to_bool(pred.get("has_anomaly"))
        gt_flag = _to_bool(gt.get("has_anomaly"))
        anomaly_correct = (pred_flag is not None and pred_flag == gt_flag)

        gt_objects = gt.get("objects") or []
        pred_objects = pred.get("objects") or []

        if gt_flag is True:
            # ---- 正样本：漏检重罚 ----
            if anomaly_correct:
                reward += self.W_BINARY_POS
                if str(pred.get("anomaly_type", "")).strip() == \
                   str(gt.get("anomaly_type", "")).strip():
                    reward += self.W_TYPE_POS
            reward += self.W_BBOX_POS * _detection_f1(
                pred_objects, gt_objects, iou_thr=0.5
            )
        else:
            # ---- 负样本：正确判定正常给大头，乱框会扣 bbox ----
            if anomaly_correct:
                reward += self.W_BINARY_NEG
            # 负样本 GT 同样带设备框：必须正确检测设备位置才能拿 bbox 分
            reward += self.W_BBOX_NEG * _detection_f1(
                pred_objects, gt_objects, iou_thr=0.5
            )

        return max(0.0, min(1.0, reward))


class SceneAnomalyReward(ORM):
    REQUIRED_FIELDS = [
        "scene_description", "device_status", "detection_analysis",
        "has_anomaly", "anomaly_type", "confidence", "reason", "objects",
    ]
    DESC_FIELDS = ["scene_description", "device_status", "detection_analysis", "reason"]

    W_FORMAT = 0.08
    W_TEXT   = 0.07

    # 正样本（gt 有异常）：漏检重罚
    W_BINARY_POS = 0.50
    W_TYPE_POS   = 0.10
    W_BBOX_POS   = 0.25

    # 负样本（gt 正常）：误报轻罚；检测设备位置是核心任务
    W_BINARY_NEG = 0.25
    W_BBOX_NEG   = 0.60


    def __call__(self, completions, task=None, solution=None, **kwargs):
        tasks = task if task is not None else kwargs.get("task", [None] * len(completions))
        solutions = solution if solution is not None else kwargs.get("solution", [None] * len(completions))
        rewards = []
        for completion, t, sol in zip(completions, tasks, solutions):
            if not (isinstance(t, str) and t.startswith("scene_")):
                rewards.append(None)
                continue
            rewards.append(self._score(completion, sol))
        return rewards

    def _score(self, completion, solution):
        pred = _extract_json(completion)
        gt = _extract_json(solution) if isinstance(solution, str) else solution
        if pred is None or gt is None:
            return 0.0

        reward = 0.0

        # 1. 格式分
        fields_ok = all(f in pred for f in self.REQUIRED_FIELDS)
        bool_ok = _to_bool(pred.get("has_anomaly")) is not None
        conf_ok = pred.get("confidence") in ("高", "中", "低")
        objects_ok = isinstance(pred.get("objects"), list)
        if fields_ok and bool_ok and conf_ok and objects_ok:
            reward += self.W_FORMAT
        elif bool_ok:
            reward += self.W_FORMAT * 0.5

        # 2. 描述质量
        reward += self.W_TEXT * _text_group_score(pred, self.DESC_FIELDS)

        # 3. 类别条件评分
        pred_flag = _to_bool(pred.get("has_anomaly"))
        gt_flag = _to_bool(gt.get("has_anomaly"))
        anomaly_correct = (pred_flag is not None and pred_flag == gt_flag)

        gt_objects = gt.get("objects") or []       # 统一按 diff 的处理
        pred_objects = pred.get("objects") or []

        if gt_flag is True:
            if anomaly_correct:
                reward += self.W_BINARY_POS
                if str(pred.get("anomaly_type", "")).strip() == \
                   str(gt.get("anomaly_type", "")).strip():
                    reward += self.W_TYPE_POS
            reward += self.W_BBOX_POS * _detection_f1(
                pred_objects, gt_objects, iou_thr=0.5
            )
        else:
            if anomaly_correct:
                reward += self.W_BINARY_NEG
            reward += self.W_BBOX_NEG * _detection_f1(
                pred_objects, gt_objects, iou_thr=0.5
            )

        return max(0.0, min(1.0, reward))


class PersonDetReward(ORM):
    """人形检测：persondet"""

    # 权重合计 1.0
    W_FORMAT = 0.10
    W_PUREJSON = 0.05
    W_CONSIST = 0.10
    W_COUNT = 0.30
    W_BBOX_F1 = 0.25
    W_BBOX_IOU = 0.20

    def __call__(self, completions, task=None, solution=None, **kwargs):
        tasks = task if task is not None else kwargs.get("task", [None] * len(completions))
        solutions = solution if solution is not None else kwargs.get("solution", [None] * len(completions))

        if not isinstance(solutions, (list, tuple)):
            solutions = [solutions] * len(completions)
        if len(solutions) < len(completions):
            solutions = list(solutions) + [None] * (len(completions) - len(solutions))

        rewards = []
        for completion, t, sol in zip(completions, tasks, solutions):
            if not (isinstance(t, str) and t == "persondet"):
                rewards.append(None)
                continue

            pred = _extract_json(completion)
            gt = _extract_json(sol)
            if pred is None or gt is None:
                rewards.append(0.0)
                continue

            score = 0.0
            pred_objs = pred.get("objects", [])
            gt_objs = gt.get("objects", [])
            valid_preds = _parse_objects(pred_objs)
            valid_gts = _parse_objects(gt_objs)

            # 1. 格式分
            fmt = 0.0
            if "person_count" in pred:
                try:
                    int(pred["person_count"])
                    fmt += 0.5
                except Exception:
                    pass
            if isinstance(pred_objs, list):
                fmt += 0.5
            score += self.W_FORMAT * fmt

            # 2. 纯 JSON 约束
            if _is_pure_json(completion):
                score += self.W_PUREJSON

            # 3. 解析 person_count
            pred_count = -1
            if "person_count" in pred:
                try:
                    pred_count = int(pred["person_count"])
                except Exception:
                    pass

            gt_count = len(valid_gts)
            if "person_count" in gt:
                try:
                    gt_count = int(gt["person_count"])
                except Exception:
                    pass

            # 4. 逻辑自洽（count == 有效框数量）
            if pred_count >= 0 and pred_count == len(valid_preds):
                score += self.W_CONSIST

            # 5. 人数准确度（连续得分）
            if pred_count >= 0:
                if pred_count == 0 and gt_count == 0:
                    score += self.W_COUNT
                else:
                    err = abs(pred_count - gt_count) / max(1, max(pred_count, gt_count))
                    score += self.W_COUNT * max(0.0, 1.0 - err)

            # 6. BBox 质量：F1 + mean IoU
            f1, mean_iou = _detection_f1_and_iou(pred_objs, gt_objs, iou_thr=0.5, use_label=True)
            score += self.W_BBOX_F1 * f1
            score += self.W_BBOX_IOU * mean_iou

            rewards.append(max(0.0, min(1.0, float(score))))
        return rewards


# ============================================================
# 第三部分：人员行为识别（personaction）及专用工具函数
# ============================================================

_ACTION_ALLOWED_LABELS = frozenset("ABCDEFGH")


def _to_int_seconds(v):
    """将 start_time 转为 int 秒，失败返回 None。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            return int(float(s)) if s else None
    except Exception:
        return None


def _parse_action_labels(raw):
    """解析 labels 字段为有序列表；不合法返回 None。"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    parts = [x.strip().upper() for x in raw.split(",")]
    if not parts or any(p not in _ACTION_ALLOWED_LABELS for p in parts):
        return None
    return parts


def _canonicalize_labels(labels):
    """标准化：去重、排序、合并为字符串。"""
    if not isinstance(labels, list) or not labels:
        return None
    return ",".join(sorted(set(labels)))


def _is_valid_action_labels(labels):
    """检查 labels 是否满足业务规则。"""
    if not isinstance(labels, list) or not labels:
        return False
    label_set = set(labels)
    if any(x not in _ACTION_ALLOWED_LABELS for x in label_set):
        return False
    if len(set(labels)) != len(labels):  # 有重复
        return False
    if labels != sorted(labels):  # 未按字母排序
        return False
    if "H" in label_set and len(label_set) > 1:  # H 不独占
        return False
    if "A" in label_set and "D" not in label_set:  # A => D
        return False
    return True


def _weighted_label_f1(pred_labels, gt_labels, weights):
    """加权多标签 F1 ∈ [0, 1]。"""
    if pred_labels is None or gt_labels is None:
        return 0.0
    pred_set = set(pred_labels)
    gt_set = set(gt_labels)
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0

    tp_w = sum(weights.get(x, 1.0) for x in (pred_set & gt_set))
    pred_w = sum(weights.get(x, 1.0) for x in pred_set)
    gt_w = sum(weights.get(x, 1.0) for x in gt_set)

    if pred_w <= 0 or gt_w <= 0:
        return 0.0
    p = tp_w / pred_w
    r = tp_w / gt_w
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def _time_score(pred_time, gt_time, gt_labels):
    """start_time 奖励 ∈ [0, 1]。
    - H 场景：pred_time == 0 满分，否则 0.5（不完全惩罚）
    - 有异常场景：阶梯式容差
    """
    if pred_time is None or gt_time is None:
        return 0.0

    gt_set = set(gt_labels) if gt_labels else set()

    if gt_set == {"H"}:
        return 1.0 if pred_time == 0 else 0.5  # H 场景不强制为0，宽松处理

    diff = abs(pred_time - gt_time)
    if diff == 0:   return 1.0
    if diff <= 1:   return 0.9
    if diff <= 3:   return 0.7
    if diff <= 5:   return 0.4
    return 0.0


class PersonActionReward(ORM):
    """人员行为识别：personaction"""

    W_FORMAT = 0.05
    W_PUREJSON = 0.05
    W_LABEL_RULE = 0.10
    W_LABEL_WEIGHTED_F1 = 0.60
    W_LABEL_EXACT = 0.15
    W_TIME = 0.05

    LABEL_WEIGHTS = {
        "A": 2.5, "B": 2.5,
        "C": 1.2, "D": 1.2,
        "E": 0.8, "F": 0.8,
        "G": 3.0,
        "H": 0.8,
    }

    def __call__(self, completions, task=None, solution=None, **kwargs):
        tasks = task if task is not None else kwargs.get("task", [None] * len(completions))
        solutions = solution if solution is not None else kwargs.get("solution", [None] * len(completions))

        if not isinstance(solutions, (list, tuple)):
            solutions = [solutions] * len(completions)
        if len(solutions) < len(completions):
            solutions = list(solutions) + [None] * (len(completions) - len(solutions))

        label_weights = kwargs.get("personaction_label_weights", self.LABEL_WEIGHTS)
        if not isinstance(label_weights, dict):
            label_weights = self.LABEL_WEIGHTS

        rewards = []
        for completion, t, sol in zip(completions, tasks, solutions):
            if not (isinstance(t, str) and t == "personaction"):
                rewards.append(None)
                continue

            pred = _extract_json(completion)
            gt = _extract_json(sol)
            if pred is None or gt is None:
                rewards.append(0.0)
                continue

            score = 0.0

            # 1. 格式
            fmt = 0.0
            if "labels" in pred and isinstance(pred.get("labels"), str):
                fmt += 0.5
            if "start_time" in pred and _to_int_seconds(pred.get("start_time")) is not None:
                fmt += 0.5
            score += self.W_FORMAT * fmt

            # 2. 纯 JSON
            if _is_pure_json(completion):
                score += self.W_PUREJSON

            # 3. 解析
            pred_labels = _parse_action_labels(pred.get("labels"))
            gt_labels = _parse_action_labels(gt.get("labels"))
            if gt_labels is None:
                raw = str(gt.get("labels", "")).strip().upper()
                if raw == "H":
                    gt_labels = ["H"]

            pred_time = _to_int_seconds(pred.get("start_time"))
            gt_time_parsed = _to_int_seconds(gt.get("start_time"))   # 不再兜底成 0

            # 4. 规则合法性
            rule_valid = (pred_labels is not None
                          and _is_valid_action_labels(pred_labels))
            if rule_valid:
                score += self.W_LABEL_RULE

            # 5. 加权 F1（第 12 点：违反业务规则时打折）
            label_f1 = _weighted_label_f1(pred_labels, gt_labels, label_weights)
            if pred_labels is not None and not rule_valid:
                label_f1 *= 0.6   # 硬约束折扣
            score += self.W_LABEL_WEIGHTED_F1 * label_f1

            # 6. 完全正确额外奖励（第 2 点：避免 None==None 白送分）
            if (pred_labels is not None
                    and _canonicalize_labels(pred_labels)
                    == _canonicalize_labels(gt_labels)):
                score += self.W_LABEL_EXACT

            # 7. 时间保持项（第 10 点：gt 无时间标注时该项跳过，避免误罚）
            if gt_time_parsed is None:
                effective_t = 1.0    # 无 GT 视为满分，等价于该项被跳过
            else:
                t_score = _time_score(pred_time, gt_time_parsed, gt_labels)
                effective_t = t_score * (0.3 + 0.7 * label_f1)
            score += self.W_TIME * effective_t

            rewards.append(max(0.0, min(1.0, float(score))))
        return rewards


# ============================================================
# 第四部分：统一注册
# ============================================================

orms["device_diff_reward"] = DeviceDiffReward
orms["scene_anomaly_reward"] = SceneAnomalyReward
orms["person_det_reward"] = PersonDetReward
orms["person_action_reward"] = PersonActionReward