def compute_accuracy(outputs, y):
    
    if len(y.shape) == 2:
        lm_targets = y
    else:
        lm_targets = y[:, 1:, 0]

    lm_preds = outputs["lm_scores"].max(dim=-1)[1]
    lm_acc = ((lm_preds == lm_targets) * (lm_targets > 6)).sum(dim=1) == (lm_targets > 6).sum(
        dim=1
    )
    if "span_b_scores" in outputs:
        sb_targets = y[:, 1:, 1]
        sb_preds = outputs["span_b_scores"].max(dim=-1)[1]
        sb_acc = ((sb_preds == sb_targets) * (sb_targets >= 0)).sum(dim=1) == (
            sb_targets >= 0
        ).sum(dim=1)
        se_targets = y[:, 1:, 2]
        se_preds = outputs["span_e_scores"].max(dim=-1)[1]
        se_acc = ((se_preds == se_targets) * (se_targets >= 0)).sum(dim=1) == (
            se_targets >= 0
        ).sum(dim=1)
        sp_acc = sb_acc * se_acc
        full_acc = lm_acc * sp_acc
        if "text_span_start_scores" in outputs:
            text_span_b_targets = y[:, 1:, 3]
            text_span_e_targets = y[:, 1:, 4]
            text_span_b_pred = outputs["text_span_start_scores"].max(dim=-1)[1]
            text_span_e_pred = outputs["text_span_end_scores"].max(dim=-1)[1]
            text_span_b_acc = (
                (text_span_b_pred == text_span_b_targets) * (text_span_b_targets >= 0)
            ).sum(dim=1) == (text_span_b_targets >= 0).sum(dim=1)
            text_span_e_acc = (
                (text_span_e_pred == text_span_e_targets) * (text_span_e_targets >= 0)
            ).sum(dim=1) == (text_span_e_targets >= 0).sum(dim=1)
            text_span_acc = text_span_b_acc * text_span_e_acc
            return (lm_acc, sp_acc, text_span_acc, full_acc)
        else:
            return (lm_acc, sp_acc, full_acc)
    else:
        return lm_acc