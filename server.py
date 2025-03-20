import  flwr    as  fl
import  numpy   as  np

def aggregate_median(weights_list):
    """Aggregate client weights usingg the median"""
    aggregated_weights  =   []
    for weights in  zip(*weights_list):
        stacked =   np.stack(weights,   axis=0)
        median_weights  =   np.median(stacked,  axis=0)
        aggregated_weights.append(median_weights)
    return  aggregated_weights

class   MedianStrategy(fl.server.strategy.FedAvg):
    """Custom Strategy that uses median aggregation"""

    def aggregate_fit(self, rnd,    results,    failures):
        if  not results:
            return  None,   {}
        weights_results  =   [parameters for _,  parameters, _   in  results]
        aggregated_weights  =   aggregate_median(weights_results)
        return  aggregated_weights, {}

#Start the federated server
if  __name__    ==  "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=MedianStrategy(),
        config=fl.server.ServerConfig(num_rounds=5)
    )