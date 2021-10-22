
def load_model():
    from transformers import BertTokenizer
    from model import BertForMultiLabelClassification
    from multilabel_pipeline import MultiLabelPipeline
    from pprint import pprint

    tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

    goemotions = MultiLabelPipeline(
        model=model,
        tokenizer=tokenizer,
        threshold=0.3
    )
    return goemotions