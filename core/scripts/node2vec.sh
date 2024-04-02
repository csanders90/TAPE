python wb_tune2.py --cfg core/configs/arxiv_2023/node2vec.yaml --sweep core/configs/arxiv_2023/sweep.yaml

python wb_tune2.py --cfg core/configs/cora/node2vec.yaml --sweep core/configs/cora/sweep.yaml

python wb_tune2.py --cfg core/configs/pubmed/node2vec.yaml --sweep core/configs/pubmed/sweep.yaml



# Today 
python wb_tune2.py --cfg core/configs/cora/node2vec.yaml --sweep core/configs/cora/sweep2.yaml

python wb_tune2.py --cfg core/configs/arxiv_2023/node2vec.yaml --sweep core/configs/arxiv_2023/sweep2.yaml


# Evening
python wb_tune2.py --cfg core/configs/cora/node2vec.yaml --sweep core/configs/cora/sweep3.yaml


python wb_tune2.py --cfg core/configs/arxiv_2023/node2vec.yaml --sweep core/configs/arxiv_2023/sweep3.yaml

# deepwalk 
python wb_tune2.py --cfg core/configs/arxiv_2023/deepwalk.yaml --sweep core/configs/arxiv_2023/dw_sweep1.yaml

python wb_tune2.py --cfg core/configs/cora/deepwalk.yaml --sweep core/configs/cora/dw_sweep1.yaml

python wb_tune2.py --cfg core/configs/pubmed/deepwalk.yaml --sweep core/configs/pubmed/dw_sweep1.yaml


# second round deepwalk
# deepwalk 
python wb_tune2.py --cfg core/configs/arxiv_2023/deepwalk.yaml --sweep core/configs/arxiv_2023/dw_sweep2.yaml

python wb_tune2.py --cfg core/configs/cora/deepwalk.yaml --sweep core/configs/cora/dw_sweep2.yaml

python wb_tune2.py --cfg core/configs/pubmed/deepwalk.yaml --sweep core/configs/pubmed/dw_sweep2.yaml

