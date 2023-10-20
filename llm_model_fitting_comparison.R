require(ggplot2)
require(GGally)
require(reshape2)
require(lme4)
require(compiler)
require(parallel)
require(boot)
require(lattice)
library(glmnet)
library(paletteer)
library(ggpattern)

eval_func <- function(model, x, y) {
  y_test_pred <- predict(model, x, type = "response") > 0.5
  sum(y_test_pred == (y==1))/length(y)
}

# load data and transform variables
df_train = read.csv("~/OneDrive/projects/prediction/mood_management/df_train.csv")
df_test = read.csv("~/OneDrive/projects/prediction/mood_management/df_test.csv")
df_train$participant <- as.character(df_train$participant)
df_test$participant <- as.character(df_test$participant)
df_train$key <- as.factor(df_train$key)
df_test$key <- as.factor(df_test$key)
df_test$valence_diff <- (df_test$valence_diff - mean(df_train$valence_diff))/sd(df_train$valence_diff)
df_train$valence_diff <- (df_train$valence_diff - mean(df_train$valence_diff))/sd(df_train$valence_diff)
df_test$arousal_diff <- (df_test$arousal_diff - mean(df_train$arousal_diff))/sd(df_train$arousal_diff)
df_train$arousal_diff <- (df_train$arousal_diff - mean(df_train$arousal_diff))/sd(df_train$arousal_diff)
df_test$age <- (df_test$age - mean(df_train$age))/sd(df_train$age)
df_train$age <- (df_train$age - mean(df_train$age))/sd(df_train$age)

df_train$gender <- factor(df_train$gender, levels = c("Female", "Male", "Gender queer/Gender nonconforming",
                                                      "Transgender", "Would prefer not to answer"))

df_test$gender <- factor(df_test$gender, levels = c("Female", "Male", "Gender queer/Gender nonconforming",
                                                    "Transgender", "Would prefer not to answer"))

df_train$race <- factor(df_train$race, levels = c("Caucasian (or White)", "African American (or Black)", "Mixed race or ethnicity",  
                                                  "Asian American (or Pacific Islander)", "Prefer not to answer", "Hispanic American (or Latino)",   
                                                  "Native American (or American Indian)", "Hispanic", "African"))
df_test$race <- factor(df_test$race, levels = c("Caucasian (or White)", "African American (or Black)", "Mixed race or ethnicity",  
                                                "Asian American (or Pacific Islander)", "Prefer not to answer", "Hispanic American (or Latino)",   
                                                "Native American (or American Indian)", "Hispanic", "African"))




## Model Fitting
# valence model
m_valence <- glm(key ~ valence_diff, data = df_train,
                 family = binomial)
p_arousal <- eval_func(m_valence, df_test, df_test$key)
e_arousal <- eval_func(m_valence, df_train, df_train$key)

# valence model
m_arousal <- glm(key ~ arousal_diff, data = df_train,
                 family = binomial)
p_valence <- eval_func(m_arousal, df_test, df_test$key)
e_valence <- eval_func(m_arousal, df_train, df_train$key)


# valence arousal model
m_valence_arousal <- glm(key ~ valence_diff + arousal_diff, data = df_train,
                         family = binomial)
p_valence_arousal <- eval_func(m_valence_arousal, df_test, df_test$key)
e_valence_arousal <- eval_func(m_valence_arousal, df_train, df_train$key)

# mood
m_mood <- glm(key ~ valence_diff*mood_valence + arousal_diff*mood_arousal, data = df_train,
              family = binomial)
p_mood <- eval_func(m_mood, df_test, df_test$key)
e_mood <- eval_func(m_mood, df_train, df_train$key)

# age
m_age <- glm(key ~ valence_diff*age + arousal_diff*age, data = df_train,
             family = binomial)
p_age <- eval_func(m_age, df_test, df_test$key)
e_age <- eval_func(m_age, df_train, df_train$key)

# race
m_race <- glm(key ~ valence_diff*race + arousal_diff*race, data = df_train,
              family = binomial)
p_race <- eval_func(m_race, df_test, df_test$key)
e_race <- eval_func(m_race, df_train, df_train$key)

# gender
m_gender <- glm(key ~ valence_diff*gender + arousal_diff*gender, data = df_train,
                family = binomial)
p_gender <- eval_func(m_gender, df_test, df_test$key)
e_gender <- eval_func(m_gender, df_train, df_train$key)

# two way linear model
m_linear <- glm(key ~ valence_diff*mood_valence + valence_diff*age + valence_diff*race + valence_diff*gender +
                  arousal_diff*mood_arousal + arousal_diff*age + arousal_diff*race + arousal_diff*gender, data = df_train,
                family = binomial)
p_linear <- eval_func(m_linear, df_test, df_test$key)
e_linear <- eval_func(m_linear, df_train, df_train$key)


# 7 way interaction model
m_7way <- glm(key ~ valence_diff*arousal_diff*mood_valence*mood_arousal*age*race*gender, data = df_train,
              family = binomial)
p_7way <- eval_func(m_7way, df_test, df_test$key)
e_7way <- eval_func(m_7way, df_train, df_train$key)

# lasso model
x <- model.matrix(key ~ valence_diff*arousal_diff*mood_valence*mood_arousal*age*race*gender,
                  df_train[(df_train$race %in% unique(df_test$race))&
                             (df_train$gender %in% unique(df_test$gender)),])[,-1]
y <- df_train$key[(df_train$race %in% unique(df_test$race))&
                    (df_train$gender %in% unique(df_test$gender))]
x_test <- model.matrix(key ~ valence_diff*arousal_diff*mood_valence*mood_arousal*age*race*gender,
                       df_test)[,-1]
y_test <- df_test$key
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
m_lasso <- glmnet(x, y, family = "binomial", alpha = 1, lambda = cv.lasso$lambda.min)
p_lasso <- eval_func(m_lasso, x_test, df_test$key)
e_lasso <- eval_func(m_lasso, x, df_train$key[(df_train$race %in% unique(df_test$race))&
                                                (df_train$gender %in% unique(df_test$gender))])



# SVM model is fit in python using sklearn packages
# Check the jupyter notebook "svm_model_fitting" for code to fit the model and evaluations

p_svm <- 0.634
e_svm <- 0.653
model_names <- c("movie", "movie x mood", 
                 "movie x race", "movie x gender", "movie x age", "movie x (mood + user)", 
                 "movie x mood x user", 
                 "Ridge",
                 "SVM")
model_cat <- c(rep("simple model", 5),
               "theoretical model", "complex model", 
               "ridge complex model", "black box model")
evaluate_short_df <- as.data.frame(list("predictability"=c(p_valence_arousal, p_mood,  p_race, p_gender, 
                                                           p_age, p_linear, p_7way, p_lasso, p_svm), 
                                        "explanatbility" = c(e_valence_arousal, e_mood, e_race, e_gender, e_age,
                                                             e_linear, e_7way, e_lasso, e_svm),
                                        "model" = model_names))
evaluate_short_df

evaluate_model_df <- as.data.frame(list("value"=c(p_valence_arousal, p_mood,  p_race, p_gender, 
                                                  p_age, p_linear, p_7way, p_lasso, p_svm, 
                                                  e_valence_arousal, e_mood, e_race, e_gender, e_age,
                                                  e_linear, e_7way, e_lasso, e_svm),
                                        "Evaluation"=c(rep("predictability", 9), rep("explainability", 9)),
                                        "models" = rep(model_names, 2),
                                        "Model_category" = rep(model_cat, 2)))
evaluate_model_df$Model_category <- factor(evaluate_model_df$Model_category,
                                           levels = c("simple model", "theoretical model", "complex model",
                                                      "ridge complex model", "black box model"))


plt <- evaluate_model_df %>% ggplot(aes(x=value, y=reorder(models, -(1:18)),
                                        pattern=Evaluation,
                                        fill=Model_category)) +
  geom_bar_pattern(stat="identity", position = "dodge", width=.8,
                   color = "black", 
                   pattern_fill = "black",
                   pattern_angle = 45,
                   pattern_density = 0.1,
                   pattern_spacing = 0.025,
                   pattern_key_scale_factor = 0.6) + 
  scale_pattern_manual(values = c(predictability = "stripe", explainability = "none")) +
  coord_cartesian(xlim = c(0.5, 0.65)) +
  theme(
    axis.text.y = element_blank(),
    panel.background = element_rect(fill = "white"),
    panel.grid.major.x = element_line(color = "#A8BAC4", size = 0.3),
    axis.ticks.length = unit(0, "mm"),
    axis.line.y.left = element_line(color = "black"),
    axis.text.x = element_text(family = "Econ Sans Cnd", size = 16)
  ) + 
  geom_text(
    aes(x=0.2, y = reorder(models, -(1:18)), label = reorder(models, -(1:18))),
    hjust = 0,
    nudge_x = 0.3,
    colour = "white",
    family = "Econ Sans Cnd",
    size = 6
  ) + labs(
    title = "Model Comparison for Different Models",
    subtitle = "Predicting out-of-sample data & Estimating in-sample data",
    x = "Accuracy",
    y = "Model"
  ) + 
  theme(
    plot.title = element_text(
      family = "Econ Sans Cnd", 
      face = "bold",
      size = 16
    ),
    axis.title = element_text(
      family = "Econ Sans Cnd", 
      size = 14
    ),
    legend.text=element_text(size=12),
    legend.title=element_text(size=14)
  ) +
  scale_fill_paletteer_d("calecopal::lupinus")

plt

ggsave("~/OneDrive/projects/prediction/mood_management/model_comparison.png",
       dpi=300)