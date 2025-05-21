ann_res = [f"{class_vocab[i]}: {ann_pred[i] * 100:.2f}%" for i in np.argsort(ann_pred)[::-1]]
    lstm_res = [f"{class_vocab[i]}: {lstm_pred[i] * 100:.2f}%" for i in np.argsort(lstm_pred)[::-1]]
    gru_res = [f"{class_vocab[i]}: {gru_pred[i] * 100:.2f}%" for i in np.argsort(gru_pred)[::-1]]

    return ann_res, lstm_res, gru_res