## Copyright (C) 2020 by Landmark Acoustics LLC

#' Build a List of Formulae from Combinations of Terms
#'
#' If you want to compare a bunch of models to select the best one then it is
#' helpful to quickly generate every possible model that combines your terms.
#'
#' @param form The formula that will get extra terms
#' @param extra.terms The additional terms that will be added with tail
#' recursion.
#' @return A list of formulae that includes every combination of `model` and
#' each item in `extra.terms`.
#' @importFrom stats as.formula update
#' @export
recurrent.update <- function(form, extra.terms){
    new.form <- update(form, as.formula(paste('. ~ . +', extra.terms[1])))
    result = list(new.form)
    if(length(extra.terms) > 1){
        result = c(result,
                   recurrent.update(new.form,
                                    extra.terms[-1]),
                   recurrent.update(form,
                                    extra.terms[-1]))
    }
    return(result)
}
