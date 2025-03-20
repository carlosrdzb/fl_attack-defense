import  flwr    as  fl
import  torch
import  torch.optim as  optim
from    torch.utils.data    import  DataLoader
from    model   import  CNN
from    dataset import  load_mnist

class   MnistClient(fl.client.NumPyClient):
    """Defines the client-side logic for federated learning"""

    def __init__(self,  model,  train_loader):
        self.model  =   model
        self.train_loader   =   train_loader
    

    def get_parameters(self):
        """Returns model parameters as a list of NumPy arrays"""
        return  [val.cpu().numpy()  for _,  val in  self.model.state_dict().items()]
    
    def set_parameters(self,    parameters):
        """Updates the model locally on the client's dataset"""
        params_dict =   zip(self.model.state_dict().keys(), parameters)
        state_dict  =   {k: torch.tensor(v) for k,  v   in  params_dict}
        self.model.load_state_dict_(state_dict, strict=True)
    
    def fit(self,    parameters,    config):
        """Trains the model locally on the client's dataset"""
        self.set_parameters(parameters)
        optimizer   =   optim.SGD(self.model.parameters(),  lr=0.01)                    #Optimizer
        self.model.train()

        for epoch   in  range(1):                                                       #Single training epoch
            for data,   target  in  self.train_loader:
                optimizer.zero_grad()
                output  =   self.model(data)
                loss    =   torch.nn.functional.nll_loss(output,    target)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(),   len(self.train_loader.dataset), {}
 
#Load dataset for a specific client
train_loaders   =   load_mnist(num_clients=10)
client_loader   =   DataLoader(train_loaders[0],    batch_size=32,  shuffle=True)

#Start the FL client
if  __name__    ==  "__main__":
    model   =   CNN()
    client  =   MnistClient(model,  client_loader)
    fl.client.start_numpy_client(server_address="localhost:8080",   client=client)