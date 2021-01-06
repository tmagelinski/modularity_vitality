import gzip
import igraph as ig


def readPAData(file):
    edges = set()
    with gzip.open(file, 'rb') as f:
        lines = f.readlines()
        for line in lines[4:]:
            source, target = line.split()
            edge = tuple(sorted([int(source), int(target)]))
            edge = (str(edge[0]), str(edge[1]))
            edges.add(edge)
    edges = list(edges)
    assert(len(edges) == 1541898)

    sources = [e[0] for e in edges]
    targets = [e[1] for e in edges]
    nodes = set(sources)
    nodes.update(targets)
    nodes = list(nodes)
    assert(len(nodes) == 1088092)
    return nodes, edges


def makePANet():
    raw_dir = './data/external/'
    processed_dir = './data/processed/'
    raw_filename = 'roadNet-PA.txt.gz'
    processed_filename = 'roadNet-PA.pkl.gz'
    nodes, edges = readPAData(raw_dir + raw_filename)
    g = ig.Graph()
    g.add_vertices(nodes)
    g.add_edges(edges)
    lc = g.subgraph(max(g.components(), key=len))
    assert(g.vcount() == 1088092)
    assert(g.ecount() == 1541898)
    assert(lc.vcount() == 1087562)
    assert(lc.ecount() == 1541514)
    g.write_picklez(processed_dir + processed_filename)
    return g


def main():
    makePANet()


if __name__ == '__main__':
    main()


# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
