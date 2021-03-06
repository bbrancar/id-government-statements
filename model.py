from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

def classification_model(config_data):
    tokenizer = AutoTokenizer.from_pretrained(config_data['model']['transformer_name'], add_prefix_space=True)
    config_model = AutoConfig.from_pretrained(config_data['model']['transformer_name'])
    config_model.num_labels = 2
    # TODO: add new config_model keys, take the input from respective keys of config_data.json
    #
    transformer_model = AutoModelForSequenceClassification.from_pretrained(config_data['model']['transformer_name'], config=config_model)

    return transformer_model, tokenizer, config_model