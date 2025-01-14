
import statsmodels.formula.api as smf
import pandas as pd

def forward_selection(data, response, significance_level=0.05):
    """
    data               : DataFrame contendo a variável resposta e todas as candidatas (X).
    response           : string com o nome da variável dependente (y).
    significance_level : valor de corte para p-valor na inclusão de cada variável.
    
    Retorna:
    - model: o modelo final ajustado (objeto statsmodels RegressionResults).
    - selected_vars: lista das variáveis selecionadas.
    """

    # Separamos as colunas exceto a de resposta
    remaining_vars = set(data.columns)
    remaining_vars.remove(response)
    
    # Começamos sem nenhuma variável
    selected_vars = []
    
    # Enquanto estiver encontrando variáveis com p-valor significativo, continua
    changed = True
    
    while changed and len(remaining_vars) > 0:
        changed = False
        
        # Para cada variável ainda não selecionada, testamos adicioná-la no modelo
        pvals = []
        for candidate in remaining_vars:
            # Montamos a fórmula, que é "response ~ variaveis_selecionadas + candidato"
            formula = f"{response} ~ {' + '.join(selected_vars + [candidate])}" if selected_vars \
                      else f"{response} ~ {candidate}"
            
            # Ajustamos o modelo
            model = smf.ols(formula, data=data).fit()
            
            # Coletamos o p-valor do candidato
            # (o nome do parâmetro no pvalues será igual ao nome da variável candidata 
            #  ou 'Intercept' quando for o intercepto)
            pvals.append((candidate, model.pvalues[candidate]))
        
        # Ordenamos as variáveis candidatas por menor p-valor
        pvals.sort(key=lambda x: x[1])
        
        # Pegamos a candidata com menor p-valor
        best_candidate, best_pval = pvals[0]
        
        # Se o melhor p-valor for menor que o nível de significância, incluímos a variável
        if best_pval < significance_level:
            selected_vars.append(best_candidate)
            remaining_vars.remove(best_candidate)
            changed = True
    
    # Ao final, ajustamos o modelo definitivo com as variáveis selecionadas
    formula = f"{response} ~ {' + '.join(selected_vars)}" if selected_vars \
              else f"{response} ~ 1"  # Se nenhuma variável foi selecionada
    final_model = smf.ols(formula, data=data).fit()
    
    return final_model, selected_vars


