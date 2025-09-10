#process_dog_data.py

import yaml
from pathlib import Path
import random 
import json
def get_backbone_dogs(cfg, all_gids):
    '''
    Return the global ids of the backbone dogs chosen 
    '''
    num_pretrain_dogs = cfg['num_pretrain_dogs']
    seed = cfg['random_seed']

    if num_pretrain_dogs > len(all_gids):
        raise ValueError(f"num_pretrain_dogs={num_pretrain_dogs} > total classes={len(all_gids)}")
    
    all_gids_copy = list(all_gids)
    rng = random.Random(seed)
    rng.shuffle(all_gids_copy)
    return all_gids_copy[:num_pretrain_dogs]

def create_tasks(cfg, all_gids, backbone_training_dogs):
    '''
    Create a CIL schedule: each task introduces K *new* breeds,
    excluding the backbone breeds.
    Returns a list of lists: [[new_gids_task0], [new_gids_task1], ...]
    
    '''
    K = cfg['num_cil_dogs_per_task']
    num_tasks = cfg['num_tasks']
    seed = cfg['random_seed']
    
    all_gids_copy = list(all_gids)
    rng = random.Random(seed)
    rng.shuffle(all_gids_copy)
    
    backbone_set = set(backbone_training_dogs)
    remaining = [g for g in all_gids_copy if g not in backbone_set]
    
    if K*num_tasks > len(remaining):
        raise ValueError(f"Need {K*num_tasks} new dogs, given {len(remaining)} after backbone training")

    return [remaining[i*K:(i+1)*K] for i in range(num_tasks)]


##---------- MAIN FUNCTION
def process_dog_data(cfg):
    
    """Create task semantics"""
    print("Starting processing of dog breed data into tasks...")

    #Load indexed data
    index_json = cfg['cleaned_data_path_json']
    out_dir = Path(cfg['tasks_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    gid2breed = json.loads(Path(index_json).read_text())
    all_gids = list(map(int, gid2breed.keys()))

    
    backbone_dogs = get_backbone_dogs(cfg, all_gids)
    print("Backbone Dogs Chosen:")
    for i in backbone_dogs:
        print(f"GID: {i}, BREED: {gid2breed[str(i)]}") 
    
    schedule = create_tasks(cfg, all_gids,
                             backbone_dogs)
    
    print("CIL Schedule Created:")
    for idx, task in enumerate(schedule):
        print(f"Task {idx}, GID: {task[0]}, BREED: {gid2breed[str(task[0])]}") 
    
  
    #Save metadata/summary
    meta = {
        "dogs_per_task": cfg["num_cil_dogs_per_task"],
        "backbone_dogs": backbone_dogs,
        "schedule": schedule,
        "gid2breed": gid2breed,
        "img_size": cfg["img_size"],
        "use_bbox": cfg["use_bbox"],
        "k_new": cfg.get("k_new", None),
    }
    (out_dir / "tasks_meta.yaml").write_text(yaml.dump(meta))
    
    with open(out_dir / "tasks_human_readable.txt", "w") as f:
        f.write(f"Backbone ({len(backbone_dogs)}): " +
                ", ".join(gid2breed[str(g)] for g in backbone_dogs) + "\n\n")
        for t, gids in enumerate(schedule):
            names = [gid2breed[str(g)] for g in gids]
            f.write(f"Task {t}: {names}\n")
    
    return backbone_dogs, schedule

if __name__ == "__main__":
    process_dog_data()
