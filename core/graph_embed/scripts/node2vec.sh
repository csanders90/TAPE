python wb_tune2.py --cfg core/yamls/arxiv_2023/node2vec.yaml --sweep core/yamls/arxiv_2023/sweep.yaml

python wb_tune2.py --cfg core/yamls/cora/node2vec.yaml --sweep core/yamls/cora/sweep.yaml

python wb_tune2.py --cfg core/yamls/pubmed/node2vec.yaml --sweep core/yamls/pubmed/sweep.yaml



# Today 
python wb_tune2.py --cfg core/yamls/cora/node2vec.yaml --sweep core/yamls/cora/sweep2.yaml

python wb_tune2.py --cfg core/yamls/arxiv_2023/node2vec.yaml --sweep core/yamls/arxiv_2023/sweep2.yaml


# Evening
python wb_tune2.py --cfg core/yamls/cora/node2vec.yaml --sweep core/yamls/cora/sweep3.yaml


python wb_tune2.py --cfg core/yamls/arxiv_2023/node2vec.yaml --sweep core/yamls/arxiv_2023/sweep3.yaml

# deepwalk 
python wb_tune2.py --cfg core/yamls/arxiv_2023/deepwalk.yaml --sweep core/yamls/arxiv_2023/dw_sweep1.yaml

python wb_tune2.py --cfg core/yamls/cora/deepwalk.yaml --sweep core/yamls/cora/dw_sweep1.yaml

python wb_tune2.py --cfg core/yamls/pubmed/deepwalk.yaml --sweep core/yamls/pubmed/dw_sweep1.yaml


# second round deepwalk
# deepwalk 
python wb_tune2.py --cfg core/yamls/arxiv_2023/deepwalk.yaml --sweep core/yamls/arxiv_2023/dw_sweep2.yaml

python wb_tune2.py --cfg core/yamls/cora/deepwalk.yaml --sweep core/yamls/cora/dw_sweep2.yaml

python wb_tune2.py --cfg core/yamls/pubmed/deepwalk.yaml --sweep core/yamls/pubmed/dw_sweep2.yaml

