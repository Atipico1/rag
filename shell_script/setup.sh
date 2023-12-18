mkdir models
mkdir wikidata
wget https://docs-assets.developer.apple.com/ml-research/models/kc-ner/model.gz -O models/kc-ner-model.gz
tar -xvzf models/kc-ner-model.gz -C models/
wget https://docs-assets.developer.apple.com/ml-research/models/kc-ner/entity_info.json.gz -O wikidata/entity_info.json.gz