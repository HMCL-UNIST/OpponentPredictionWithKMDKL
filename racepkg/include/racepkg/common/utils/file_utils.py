'''
MIT License

Copyright (c) 2022 Model Predictive Control (MPC) Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import os
import pickle
import pathlib

gp_dir = os.path.expanduser('~') + '/data/'
pred_dir = os.path.join(gp_dir, 'predData/')
train_dir = os.path.join(gp_dir, 'trainingData/')
real_dir = os.path.join(gp_dir, 'realData/')
multiEval_dir = os.path.join(gp_dir, 'MultiEvalData/')
eval_dir = os.path.join(gp_dir, 'evaluationData/')
lik_dir = os.path.join(gp_dir, 'LiklihoodData/')
model_dir = os.path.join(train_dir, 'models/')
param_dir = os.path.join(gp_dir, 'params/')
track_dir = os.path.join(gp_dir, 'tracks/')
static_dir = os.path.join(gp_dir, 'statics/')
fig_dir = os.path.join(gp_dir, 'figures/')

def dir_exists(path=''):
    dest_path = pathlib.Path(path).expanduser()
    return dest_path.exists()


def create_dir(path='', verbose=False):
    dest_path = pathlib.Path(path).expanduser()
    if not dest_path.exists():
        dest_path.mkdir(parents=True)
        return dest_path
    else:
        if verbose:
            print('- The source directory %s does not exist, did not create' % str(path))
        return None


def pickle_write(data, path):
    dbfile = open(path, 'wb')
    pickle.dump(data, dbfile)
    dbfile.close()


def pickle_read(path):
    print("path = "+ str(path))
    dbfile = open(path, 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
    return data
