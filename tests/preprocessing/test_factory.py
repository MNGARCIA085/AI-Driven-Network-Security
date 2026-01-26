from net_security.preprocessors.factory import PreprocessorFactory

def test_factory_returns_nn_preprocessor(data_cfg_nn):
    prep = PreprocessorFactory.get_preprocessor("nn", data_cfg_nn)
    assert prep is not None

def test_factory_returns_tree_preprocessor(data_cfg_tree):
    prep = PreprocessorFactory.get_preprocessor("tree", data_cfg_tree)
    assert prep is not None
