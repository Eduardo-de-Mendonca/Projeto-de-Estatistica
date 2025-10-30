from models import *

def choose_prior(mle_model):
    '''
    Escolhe uma prior adequada para modelar uma variável_aleatória, com base em um modelo de MLE ajustado para a mesma variável aleatória
    '''
    assert isinstance(mle_model, Model)

    if isinstance(mle_model, BinomialModel):
        return BetaModel(1, 1) # Prior uniforme

    if USE_MLE_AS_REFERENCE_FOR_PRIORS:
        # Calcularemos a média e desvio-padrão do modelo a ajustar com base no mle
        # A média será o valor do mle, e o desvio padrão será C vezes a média
        # A desvantagem dessa abordagem é que enviesa a inferência bayesiana, fazendo com que seus resultados tenham convergência muito rápido para os do MLE e subestimem as incertezas
        C = 2000
        if isinstance(mle_model, BinomialModel):
            return BetaModel(1, 1) # Prior uniforme
        
        if isinstance(mle_model, NormalModel):
            mle_avg = mle_model.average
            # StdDev da Prior = C * Média da Prior (que é o MLE)
            prior_std_dev = abs(mle_avg) * C
            
            prior = NormalModel(average=mle_avg, std_dev=prior_std_dev)
            return prior
        
        if isinstance(mle_model, GammaModel):
            mle_rate = mle_model.beta
            average = mle_rate
            std_dev = abs(average)*C
            
            # Usamos as fórmulas para a esperança e variância de Gamma para chegar a um sistema. Resolvemos o sistema.
            a_0 = (average / std_dev)**2
            b_0 = average/(std_dev**2)
            
            prior = GammaModel(k=a_0, beta=b_0)
            return prior
        
    else:
        # A média e o desvio-padrão da prior serão 'chutados' (não levarão em consideração o MLE)
        # A média será um valor positivo pequeno, enquanto o desvio-padrão será um valor grande
        average = 0.05
        std_dev = 100
        
        if isinstance(mle_model, NormalModel):            
            prior = NormalModel(average=average, std_dev=std_dev)
            return prior
        
        if isinstance(mle_model, GammaModel):
            a_0 = (average / std_dev)**2
            b_0 = average/(std_dev**2)
            
            prior = GammaModel(k=a_0, beta=b_0)
            return prior