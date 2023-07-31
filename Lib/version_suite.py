"""Just the version of the suite, to label outputs."""
import git
import logging
import datetime
import numpy as np
from Lib._Machine import machine
from Lib._Paths import Path

version = '1.2.10'
codename = 'IceCream'

logger = logging.getLogger('ScintSuite.Version')


def exportVersion(filename)->str:
    """
    Save the version of the suite into a file

    :param filename: Name of the file to save the version

    :return: Name of the file where the version is saved
    """
    v = version.split('.')
    v1 = int(v[0])
    v2 = int(v[1])
    v3 = int(v[2])
    data = readGITcommit()
    with open(filename, 'w') as f:
        f.write('Version ID1: %i\n' % v1)
        f.write('Version ID2: %i\n' % v2)
        f.write('Version ID3: %i\n' % v3)
        f.write('Codename: %s\n' % codename)
        f.write('Branch: %s\n' % data['branch'])
        f.write('Commit: %s\n' % data['latest_comit'])
        f.write('Commit message: %s\n' % data['latest_comit_message'])
        f.write('Date: %s\n' % data['date'])
        f.write('Commit responsible: %s\n' % data['author'])
        f.write('Mail: %s\n' % data['email'])
    return filename


def readVersion(filename)->np.ndarray:
    with open(filename, 'r') as f:
        v = np.zeros(3, dtype='int')
        for i in range(3):
            v[i] = int(f.readline().split(':')[-1])
    return v


def readGITcommit()->dict:
    """
    Read the information of the latest Suite Commit

    :return out: Dictionary containing the name and author of the commit
    """
    p = Path(machine).ScintSuite
    repo = git.Repo(p)
    branch = repo.head.reference
    out = {
        'branch': branch.name,
        'latest_comit': branch.commit.hexsha,
        'latest_comit_message': branch.commit.message,
        'date': datetime.datetime.fromtimestamp(branch.commit.committed_date),
        'author': branch.commit.author.name,
        'email': branch.commit.author.email,
    }
    return out


def printGITcommit()->None:
    """
    Print the information of the latest suite commit
    :return:
    """
    data = readGITcommit()
    logger.info('Branch: %s', data['branch'])
    logger.info('Commit: %s', data['latest_comit'])
    logger.info('Commit message: %s', data['latest_comit_message'])
    logger.info('Date: %s', data['date'])
    logger.info('Commit responsible: %s', data['author'])
    logger.info('Mail: %s', data['email'])
