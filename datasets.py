"""Dataset registrations."""
import os

import numpy as np

import common


def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)


def LoadStats_badges():
    csv_file = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/stats_simplified/badges.csv'
    cols = ['Id', 'UserId', 'Date']
    return common.CsvTable('badges', csv_file, cols)


def LoadStats_comments():
    csv_file = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/stats_simplified/comments.csv'
    cols = ['Id', 'PostId', 'Score', 'CreationDate', 'UserId']
    return common.CsvTable('comments', csv_file, cols)


def LoadStats_posts():
    csv_file = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/stats_simplified/posts.csv'
    cols = ['Id', 'PostTypeId', 'CreationDate', 'Score', 'ViewCount', 'OwnerUserId', 'AnswerCount', 'CommentCount',
            'FavoriteCount', 'LastEditorUserId']
    return common.CsvTable('posts', csv_file, cols)


def LoadStats_postHistory():
    csv_file = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/stats_simplified/postHistory.csv'
    cols = ['Id', 'PostHistoryTypeId', 'PostId', 'CreationDate', 'UserId']
    return common.CsvTable('postHistory', csv_file, cols)


def LoadStats_postLinks():
    csv_file = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/stats_simplified/postLinks.csv'
    cols = ['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId']
    return common.CsvTable('postLinks', csv_file, cols)


def LoadStats_tags():
    csv_file = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/stats_simplified/tags.csv'
    cols = ['Id', 'Count', 'ExcerptPostId']
    return common.CsvTable('tags', csv_file, cols)


def LoadStats_users():
    csv_file = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/stats_simplified/users.csv'
    cols = ['Id', 'Reputation', 'CreationDate', 'Views', 'UpVotes', 'DownVotes']
    return common.CsvTable('users', csv_file, cols)


def LoadStats_votes():
    csv_file = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/stats_simplified/votes.csv'
    cols = ['Id', 'PostId', 'VoteTypeId', 'CreationDate', 'UserId']
    return common.CsvTable('votes', csv_file, cols)


# Load dataset STATS
def LoadStats(table_name):
    cols = []
    csv_file = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/stats_simplified/{}.csv'.format(table_name)
    if table_name == 'badges':
        cols = ['Id', 'UserId', 'Date']
    elif table_name == 'comments':
        cols = ['Id', 'PostId', 'Score', 'CreationDate', 'UserId']
    elif table_name == 'posts':
        cols = ['Id', 'PostTypeId', 'CreationDate', 'Score', 'ViewCount', 'OwnerUserId', 'AnswerCount', 'CommentCount',
                'FavoriteCount', 'LastEditorUserId']
    elif table_name == 'postHistory':
        cols = ['Id', 'PostHistoryTypeId', 'PostId', 'CreationDate', 'UserId']
    elif table_name == 'postLinks':
        cols = ['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId']
    elif table_name == 'tags':
        cols = ['Id', 'Count', 'ExcerptPostId']
    elif table_name == 'users':
        cols = ['Id', 'Reputation', 'CreationDate', 'Views', 'UpVotes', 'DownVotes']
    elif table_name == 'votes':
        cols = ['Id', 'PostId', 'VoteTypeId', 'CreationDate', 'UserId', 'BountyAmount']
    else:
        print("!!!!!file name error!!!!!!")

    return common.CsvTable(table_name, csv_file, cols)
