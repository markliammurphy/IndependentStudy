import torch
import torch.nn as nn

def train(example, network, criterion, optimizer, 
          gradient_norm, hidden_size):
    target_tensor, review_tensor = example
    n_words = review_tensor.size()[0]

    hidden = network.init_hidden()
    network.zero_grad()

    for i in range(n_words):
        hidden, output = network(review_tensor[i], hidden)

    loss = criterion(output, target_tensor)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(network.parameters(), 
                                   gradient_norm)
    optimizer.step()

    return output, loss.item()


def train_minibatch(batch, network, criterion, optimizer, 
                    gradient_norm, hidden_size):
    batch_target, batch_reviews, lengths = batch

    network.zero_grad()
        
    output = network(batch_reviews, lengths)

    loss = criterion(output, batch_target)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(network.parameters(), 
                                   gradient_norm)
    optimizer.step()

    return output, loss.item()


def predict(review_tensor, network, hidden_size):
    hidden = network.init_hidden(hidden_size)
    n_words = review_tensor.size()[0]
    
    for i in range(n_words):
        hidden, output = network(review_tensor[i], hidden)
        
    sigmoid = nn.Sigmoid()
    return sigmoid(output)

    
