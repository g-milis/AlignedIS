import os

def audio_generation_undetectable_exp(res_dir,eps,model_str,reweight_type,dataset_name, max_generations=None):
    from . import audio_generation as ag
    
    assert eps>=0
    assert eps<=1
    
    sub_dir=os.path.join(res_dir,dataset_name,model_str.split('/')[-1].replace('-','_'),reweight_type)
    os.makedirs(sub_dir,exist_ok=True)
    
    output_path=os.path.join(sub_dir,'audio_generation.txt')
    if eps==0:
        score_save_path=os.path.join(sub_dir,'score.txt')
    else:
        assert os.path.exists(output_path)
        eps_str=str(eps).replace('.','_')
        score_save_path=os.path.join(sub_dir,f'eps_{eps_str}.txt')

    if eps==0:
        if os.path.exists(output_path):
            print('Found exisiting output_path:', output_path)
            print('Generation skipped.')
        else:
            print("Starting generation pipeline...",flush=True)
            ag.get_output.undetectable_exp_pipeline(output_path=output_path,
                                                    model_str=model_str,
                                                    reweight_type=reweight_type,
                                                    dataset_name=dataset_name,
                                                    max_generations=max_generations)
                                                    #context_length=2)
            print("Saved generation results in:", output_path)
    
    if os.path.exists(score_save_path):
        print('Found exisiting score_save_path:')
        print(score_save_path)
        print('Score evaluation skipped.')
    else:
        print("Starting scoring pipeline...",flush=True)
        ag.evaluate_score.pipeline(
            output_path=output_path, 
            score_save_path=score_save_path,
            eps=eps,
            model_str=model_str,
            dataset_name=dataset_name,
            max_generations=max_generations
        )
        print("Saved scoring results in:", output_path)

    print('Finish audio generation.')
    return


def add_watermark_exp():
    import argparse

    parser = argparse.ArgumentParser()
    

    parser.add_argument('--model_str', type=str)
    parser.add_argument('--dataset_name',type=str,help='Dataset name for text generation')
    parser.add_argument('--reweight_type',type=str)
    parser.add_argument('--res_dir',type=str,default='results')
    parser.add_argument("--max_generations", type=int, default=None)
    
    args=parser.parse_args()

    audio_generation_undetectable_exp(res_dir=args.res_dir,
            eps=0,
            model_str=args.model_str,
            reweight_type=args.reweight_type,
            dataset_name=args.dataset_name,
            max_generations=args.max_generations)

    exit(0)


if __name__ == "__main__":
    add_watermark_exp()
