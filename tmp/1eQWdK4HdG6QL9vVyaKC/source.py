def _convert_to_batch_ts_feature_view(data, batch_first, sequence_last):
    

    if batch_first and sequence_last:
        # batch, feature, seq -> batch, seq, feature
        data = data.permute(0, 2, 1)
    elif not batch_first and not sequence_last:
        # seq, batch, feature -> batch, seq, feature
        data = data.permute(1, 0, 2)
    elif not batch_first and sequence_last:
        # feature, batch, seq -> batch, seq, feature
        data = data.permute(1, 2, 0)

    return data