def main(args):
    import os.path as osp
    from transformers import AutoModel, AutoTokenizer
    path = osp.abspath(osp.expanduser(args.path))
    fullname = osp.join(path, args.target)
    if osp.exists(fullname) and not args.force:
        print(f'{fullname} exists, use --force to overwrite it or remove it first.')
        return

    tok = AutoTokenizer.from_pretrained(args.target)
    tok.save_pretrained(fullname)
    model = AutoModel.from_pretrained(args.target)
    model.save_pretrained(fullname)
    print(f'Tokenizer {type(tok).__name__} and Model {type(model).__name__} saved to {fullname} .')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--path', '-p', default='~/.cache/complex_qa')
    parser.add_argument('target', type=str)
    main(parser.parse_args())
