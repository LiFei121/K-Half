import torch
import dgl
from GCN import GCN
from K_Half import K_Half
from sklearn.model_selection import train_test_split
import argparse
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
import time

def get_args():
    parser = argparse.ArgumentParser(description="GCN for fact classification")
    parser.add_argument('--dataset', type=str, default='ICEWS14', help='Dataset name')
    parser.add_argument('--train_file', type=str, default='train', help='train、test、valid')
    parser.add_argument('--initial_entity_emb_dim', type=int, default=128, help='Dimension of initial entity embeddings')
    parser.add_argument('--entity_out_dim_1', type=int, default=32, help='Entity output embedding dimension (layer 1)')
    parser.add_argument('--entity_out_dim_2', type=int, default=32, help='Entity output embedding dimension (layer 2)')
    parser.add_argument('--h_dim', type=int, default=32, help='Dimension of time embeddings')
    parser.add_argument('--num_ents', type=int, default=10000, help='Number of entities')
    parser.add_argument('--nheads_GAT_1', type=int, default=4, help='Number of GAT heads (layer 1)')
    parser.add_argument('--nheads_GAT_2', type=int, default=4, help='Number of GAT heads (layer 2)')
    parser.add_argument('--n_hidden', type=int, default=128, help='Hidden layer dimension for GCN')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch', type=int, default=5000, help='batch_size')
    parser.add_argument('--threshold', type=int, default=0, help='Threshold for minimum occurrences')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID to use (default: use CPU if not specified)')
    return parser.parse_args()
args = get_args()

# Check if DGL supports CUDA
def check_dgl_cuda_support():
    """Check if DGL is installed with CUDA support"""
    try:
        import dgl
        # Try to create a small graph and move it to CUDA to test
        if torch.cuda.is_available():
            test_g = dgl.graph(([], []))
            test_g.add_nodes(1)
            try:
                test_g = test_g.to(torch.device("cuda"))
                # Clean up
                del test_g
                return True
            except Exception:
                return False
        return False
    except Exception:
        return False

# Determine device
if args.gpu is not None and torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If using CUDA but DGL doesn't support it, fall back to CPU
if device.type == "cuda" and not check_dgl_cuda_support():
    print("Warning: DGL does not support CUDA. Falling back to CPU.")
    device = torch.device("cpu")

print(f"Using device: {device}")

def count_second_column_occurrences(file_path):
    with open(file_path, 'r') as file:
        second_column_data = [line.split()[1] for line in file]
    return Counter(second_column_data)

def remove_less_than_threshold(occurrences, threshold):
    return {key: value for key, value in occurrences.items() if value >= threshold}

def print_occurrences(occurrences):
    sorted_occurrences = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)
    total_occurrences = sum(occurrences.values())

    for key, value in sorted_occurrences:
        print(f"serial number: {key}, Number of occurrences: {value}")

    print(f"\n total: {total_occurrences}")

def overwrite_file_with_filtered(file_path, filtered_occurrences):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            second_column_value = line.split()[1]
            if second_column_value in filtered_occurrences:
                file.write(line)


def load_relation_data(file, train_file, emb_dim):
    valid_relation_ids = set()

    with open(train_file, 'r') as f:
        for line in f:
            relation_id = int(line.split('\t')[1])
            valid_relation_ids.add(relation_id)

    relation_dict = {}
    labels = {}

    with open(file, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            relation_semantic, relation_id, label = cols[0], int(cols[1]), int(cols[2])

            if relation_id in valid_relation_ids:
                relation_emb = torch.randn(emb_dim)

                relation_dict[relation_id] = relation_emb
                labels[relation_id] = label

    return relation_dict, labels

def load_train_data(train_file, entity_id_map):
    facts = []
    with open(train_file, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            head_entity, relation_id, tail_entity, timestamp = int(cols[0]), int(cols[1]), int(cols[2]), int(float(cols[3]))

            if head_entity in entity_id_map and tail_entity in entity_id_map:
                mapped_head = entity_id_map[head_entity]
                mapped_tail = entity_id_map[tail_entity]
                facts.append((mapped_head, relation_id, mapped_tail, timestamp))
    return facts

def create_entity_id_map(train_file):
    entity_id_map = {}
    current_index = 0
    with open(train_file, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            head_entity = int(cols[0])
            tail_entity = int(cols[2])

            if head_entity not in entity_id_map:
                entity_id_map[head_entity] = current_index
                current_index += 1
            if tail_entity not in entity_id_map:
                entity_id_map[tail_entity] = current_index
                current_index += 1

    actual_num_ents = current_index
    return entity_id_map, actual_num_ents


def construct_graph(facts):
    # Lazy import DGL
    try:
        import dgl
    except ImportError as e:
        raise ImportError(
            f"DGL library is required but could not be imported: {e}\n"
            "Please install DGL with: pip install dgl"
        )
    
    g = dgl.graph(([], []))
    g.add_nodes(len(facts))

    edges = []
    for i, (_, rel1, _, _) in enumerate(facts):
        for j, (_, rel2, _, _) in enumerate(facts):
            if i != j and rel1 == rel2:
                edges.append((i, j))

    if edges:
        src, dst = zip(*edges)
        g.add_edges(src, dst)

    return g


def generate_fact_embeddings(facts, entity_emb, relation_dict, his_temp_embs=None):
    fact_embeddings = []
    for local_idx, (head, relation_id, tail, _) in enumerate(facts):
        entity_emb_head = entity_emb[head]
        entity_emb_tail = entity_emb[tail]
        relation_emb = relation_dict[relation_id]
        time_emb = his_temp_embs[local_idx][local_idx]

        fact_emb = torch.cat([entity_emb_head, entity_emb_tail, relation_emb, time_emb], dim=0)

        fact_embeddings.append(fact_emb)

    return torch.stack(fact_embeddings)

def compute_accuracy(logits, labels):
    _, predicted = torch.max(logits, dim=1)

    correct = (predicted == labels).sum().item()

    total = labels.size(0)

    return correct / total


def write_predictions_to_file(facts, predictions, dataset, train_file, reverse_entity_id_map):

    file_path = f"data/{dataset}/{train_file}.txt"

    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        for i, line in enumerate(lines):
            line = line.strip()
            mapped_head, relation_id, mapped_tail, timestamp = facts[i]

            original_head = reverse_entity_id_map[mapped_head]
            original_tail = reverse_entity_id_map[mapped_tail]

            prediction = predictions[i]
            f.write(f"{original_head}\t{relation_id}\t{original_tail}\t{timestamp}\t{prediction}\n")

    print(f"Predictions appended to {file_path}")

def create_reverse_entity_id_map(entity_id_map):
    return {v: k for k, v in entity_id_map.items()}


def main(args):
    total_training_time = 0.0
    file_path = f'data/{args.dataset}/{args.train_file}.txt'
    occurrences = count_second_column_occurrences(file_path)# 统计每个关系的出现次数
    filtered_occurrences = remove_less_than_threshold(occurrences, args.threshold)
    print_occurrences(filtered_occurrences)
    overwrite_file_with_filtered(file_path, filtered_occurrences) #用过滤过后的数据覆盖原来的训练数据
    print(f"File filtered and overwritten: {file_path}")

    entity_id_map, actual_num_ents = create_entity_id_map(file_path) #在训练数据中统计每个实体的id map
    args.num_ents = actual_num_ents
    reverse_entity_id_map = create_reverse_entity_id_map(entity_id_map)
    # relation_dict:初始化关系的embedding，relation_labels:关系对应的长期和短期标签
    relation_dict, relation_labels = load_relation_data(f'data/{args.dataset}/relation2id.txt', file_path, args.entity_out_dim_1)

    facts = load_train_data(file_path, entity_id_map) #获取四元组
    initial_entity_emb = torch.randn(args.num_ents, args.initial_entity_emb_dim).to(device)#初始化实体embedding
    relation_dict = {k: v.to(device) for k, v in relation_dict.items()}
    entity_out_dim = [args.entity_out_dim_1, args.entity_out_dim_2]
    nheads_GAT = [args.nheads_GAT_1, args.nheads_GAT_2]
    k_half_model = K_Half(initial_entity_emb, entity_out_dim, args.h_dim, args.num_ents, nheads_GAT,
                          relation_dict=relation_dict).to(device)

    edge_list = torch.tensor([[fact[0], fact[2]] for fact in facts]).t().to(device)
    edge_type = torch.tensor([fact[1] for fact in facts]).to(device)

    all_predictions = []
    batch_size = args.batch # batch的大小
    num_batches = (len(facts) + batch_size - 1) // batch_size #计算有多少个batches
    batches = [facts[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)] #按上一行算出的批次数把 facts 列表切分成多个子列表，每个子列表就是一个批次。
    num_edges = edge_list.shape[1] # 边的数量

    print(f"\nTotal facts: {len(facts)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {num_batches}\n")

    train_losses = []
    val_losses = []
    val_accuracies = []

    for batch_idx, batch in enumerate(batches):
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_idx + 1}/{num_batches} (size: {len(batch)})")
        print(f"{'='*60}\n")
        start = batch_idx * batch_size
        end = start + len(batch)
        edge_list_batch = edge_list[:, start:end]
        edge_type_batch = edge_type[start:end]

        batch_inputs = torch.tensor([[fact[0], fact[1], fact[2], fact[3]] for fact in batch]).to(device)
        # entity_emb是经过注意力机制的实体embedding，his_temp_embs是时间的embedding
        entity_emb, his_temp_embs = k_half_model(
            Corpus_=None,
            batch_inputs=batch_inputs,
            edge_list=edge_list_batch,
            edge_type=edge_type_batch
        )

        # 对事实进行embedding
        fact_embeddings = generate_fact_embeddings(batch, entity_emb, relation_dict, his_temp_embs)
        # 长期和短期事实
        labels = torch.tensor([relation_labels[fact[1]] for fact in batch], device=device)
        # 训练集：0.8 × 0.75 = 60%；验证集：0.8 × 0.25 = 20%；测试集：20%
        train_idx, test_idx = train_test_split(list(range(len(batch))), test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)

        g = construct_graph(batch).to(device)

        gcn_model = GCN(g, in_feats=fact_embeddings.shape[1], n_hidden=args.n_hidden, n_classes=args.n_classes,
                        n_layers=args.n_layers, activation=F.relu, dropout=args.dropout).to(device)

        optimizer = torch.optim.Adam(gcn_model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            torch.cuda.empty_cache()
            start_time = time.time()
            gcn_model.train()
            logits = gcn_model(fact_embeddings)


            loss = F.cross_entropy(logits[train_idx], labels[train_idx].long())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                gcn_model.eval()
                val_logits = gcn_model(fact_embeddings)
                val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx])
                val_accuracy = compute_accuracy(val_logits[val_idx], labels[val_idx])
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            val_accuracies.append(val_accuracy)
            epoch_time = time.time() - start_time
            total_training_time += epoch_time
            print(f'Epoch {epoch}, Time: {epoch_time:.2f}s, Total: {total_training_time:.2f}s, '
                  f'Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
                  f'Val Accuracy: {val_accuracy * 100:.2f}%')

            if val_accuracy == 1.0:
                print("Validation accuracy reached 100%, stopping training early.")
                break

            #gcn_model.eval()


        with torch.no_grad():
            predictions = torch.argmax(gcn_model(fact_embeddings), dim=1)
            all_predictions.extend(predictions.tolist())

    write_predictions_to_file(facts, all_predictions, args.dataset, args.train_file, reverse_entity_id_map)
    print(f"\nTotal training time: {total_training_time:.2f} seconds")





if __name__ == "__main__":
    args = get_args()
    main(args)
