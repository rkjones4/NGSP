import os
import sys

if __name__ == '__main__':
    
    exp_name = sys.argv[1]
    cat = sys.argv[2]
    TS = int(sys.argv[3])
    args = ' '.join(sys.argv[4:])
    
    print("Training Guide Network")
    os.system(f'python3 train_guide_net.py -en {exp_name}_prop -c {cat} -ts {TS} ' + args)
    
    print("Training Likelihood Networks")
    os.system(f'python3 train_lik_models.py -en {exp_name}_{cat}_{TS}_sem -sdp lik_mods/arti_props/ -c {cat} -lm sem -ts {TS} '  + args)
    os.system(f'python3 train_lik_models.py -en {exp_name}_{cat}_{TS}_reg -sdp lik_mods/arti_props/ -c {cat} -lm reg -ts {TS} '  + args)

    print("Evaluating Models")
    os.system(f'python3 ngsp_eval.py -en {exp_name} -pmp model_output/{exp_name}_prop/prop/models/prop_net.pt -len {exp_name} -c {cat} -ts {TS} ' + args)
