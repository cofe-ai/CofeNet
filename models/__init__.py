from models.base import ExpModelBase
from models.mod_emb import ModelEMB, ModelEMB_CRF, ModelEMB_Cofe
from models.mod_cnn import ModelCNN, ModelCNN_CRF, ModelCNN_Cofe
from models.mod_lstm import ModelLSTM, ModelLSTM_CRF, ModelLSTM_Cofe
from models.mod_gru import ModelGRU, ModelGRU_CRF, ModelGRU_Cofe
from models.mod_bert import ModelBert, ModelBert_CRF, ModelBert_Cofe

from models.mod_cnn_lstm import ModelCNN_LSTM, ModelCNN_LSTM_CRF
from models.mod_bert_lstm import ModelBertLSTM, ModelBertLSTM_CRF
from models.mod_bert_cnn import ModelBertCNN, ModelBertCNN_CRF


def imp_exp_model(mod_name, mod_conf) -> ExpModelBase:
    cls = globals()[mod_name]
    return cls(mod_conf)
