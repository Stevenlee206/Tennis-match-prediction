from sklearn.naive_bayes import GaussianNB

def build_naive_bayes_model(**kwargs):
    # Accepts dynamic var_smoothing parameters to prevent variance-bound clipping.
    model_params = {
        # var_smoothing broadens the distribution curves to handle unseen variations
        'var_smoothing': kwargs.get('var_smoothing', 1e-9)
    }
    
    return GaussianNB(**model_params)