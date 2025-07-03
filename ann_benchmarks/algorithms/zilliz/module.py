from time import sleep
from pymilvus import DataType, connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import os

from ..base.module import BaseANN


def metric_mapping(_metric: str):
    _metric_type = {"angular": "COSINE", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Zilliz] Not support metric type: {_metric}!!!")
    return _metric_type


class Zilliz(BaseANN):
    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self.start_milvus()
        self.connects = connections
        max_trys = 10
        for try_num in range(max_trys):
            try:
                self.connects.connect(
                    "default",
                    host=os.environ.get('ZILLIZ_HOST', ''),
                    port=os.environ.get('ZILLIZ_PORT', '19530'),
                    user=os.environ.get('ZILLIZ_USER', ''),
                    password=os.environ.get('ZILLIZ_PASSWORD', ''),
                )
                break
            except Exception as e:
                if try_num == max_trys - 1:
                    raise Exception(f"[Zilliz] connect to milvus failed: {e}!!!")
                print(f"[Zilliz] try to connect to milvus again...")
                sleep(1)
        print(f"[Zilliz] Zilliz version: {utility.get_server_version()}")
        self.collection_name = "test_milvus"
        if utility.has_collection(self.collection_name):
            print(f"[Zilliz] collection {self.collection_name} already exists, drop it...")
            utility.drop_collection(self.collection_name)

    def start_milvus(self):
        pass

    def stop_milvus(self):
        pass

    def create_collection(self):
        filed_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True
        )
        filed_vec = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=self._dim
        )
        schema = CollectionSchema(
            fields=[filed_id, filed_vec],
            description="Test milvus search",
        )
        self.collection = Collection(
            self.collection_name,
            schema,
            consistence_level="STRONG"
        )
        print(f"[Zilliz] Create collection {self.collection.describe()} successfully!!!")

    def insert(self, X):
        # insert data
        print(f"[Zilliz] Insert {len(X)} data into collection {self.collection_name}...")
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            batch_data = X[i: min(i + batch_size, len(X))]
            entities = [
                [i for i in range(i, min(i + batch_size, len(X)))],
                batch_data.tolist()
            ]
            self.collection.insert(entities)
        self.collection.flush()
        print(f"[Zilliz] {self.collection.num_entities} data has been inserted into collection {self.collection_name}!!!")

    def get_index_param(self):
        raise NotImplementedError()

    def create_index(self):
        # create index
        print(f"[Zilliz] Create index for collection {self.collection_name}...")
        self.collection.create_index(
            field_name = "vector",
            index_params = self.get_index_param(),
            index_name = "vector_index"
        )
        utility.wait_for_index_building_complete(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        index = self.collection.index(index_name = "vector_index")
        index_progress =  utility.index_building_progress(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        print(f"[Zilliz] Create index {index.to_dict()} {index_progress} for collection {self.collection_name} successfully!!!")

    def load_collection(self):
        # load collection
        print(f"[Zilliz] Load collection {self.collection_name}...")
        self.collection.load()
        utility.wait_for_loading_complete(self.collection_name)
        print(f"[Zilliz] Load collection {self.collection_name} successfully!!!")

    def fit(self, X):
        self.create_collection()
        self.insert(X)
        self.create_index()
        self.load_collection()

    def query(self, v, n):
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids

    def done(self):
        self.collection.release()
        utility.drop_collection(self.collection_name)
        self.stop_milvus()


class ZillizFLAT(Zilliz):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self.name = f"ZillizFLAT metric:{self._metric}"

    def get_index_param(self):
        return {
            "index_type": "FLAT",
            "metric_type": self._metric_type
        }

    def query(self, v, n):
        self.search_params = {
            "metric_type": self._metric_type,
        }
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids


class ZillizIVFFLAT(Zilliz):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"ZillizIVFFLAT metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class ZillizIVFSQ8(Zilliz):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_SQ8",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"ZillizIVFSQ8 metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class ZillizIVFPQ(Zilliz):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)
        self._index_m = index_param.get("m", None)
        self._index_nbits = index_param.get("nbits", None)

    def get_index_param(self):
        assert self._dim % self._index_m == 0, "dimension must be able to be divided by m"
        return {
            "index_type": "IVF_PQ",
            "params": {
                "nlist": self._index_nlist,
                "m": self._index_m,
                "nbits": self._index_nbits if self._index_nbits else 8 
            },
            "metric_type": self._metric_type
        }
    
    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"ZillizIVFPQ metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class ZillizHNSW(Zilliz):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)

    def get_index_param(self):
        return {
            "index_type": "HNSW",
            "params": {
                "M": self._index_m,
                "efConstruction": self._index_ef
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, ef):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"ef": ef}
        }
        self.name = f"ZillizHNSW metric:{self._metric}, index_M:{self._index_m}, index_ef:{self._index_ef}, search_ef={ef}"


class ZillizSCANN(Zilliz):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "SCANN",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"ZillizSCANN metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"

class ZillizAutoindex(Zilliz):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)

    def get_index_param(self):
        return {
            "index_type": "AUTOINDEX",
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, level):
        self.search_params = {
            "params": {
                "level": level,
            }
        }
        self.name = f"ZillizAutoindex metric:{self._metric}, level:{level}"
