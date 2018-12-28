library(arules)
library(arulesViz)

a_list = list( c("a","b","c"),c("a","b"),c("a","b","d"),c("c","e"),c("a","b","d","e"))
names(a_list) = paste("Tr", c(1:5), sep = "")
a_list

trans = as(a_list, "transactions")
str(Mushroom)
summary(trans) # itemmatrix로 변환
# item 별 빈도가 나온다.
image(trans) # raw(묶음) col(itmes)


a_matrix <- matrix(c(1,2,3,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1),ncol=5)
trans2 <- as(a_matrix, "transactions")
image(trans2)

a_data.frame <- data.frame(age= as.factor(c(6,8,7,6,9,5)), grade = as.factor(c(1,3,1,1,4,1)))
a_data.frame
trans3 <- as(a_data.frame, "transactions")

summary(trans3)
summary(trans2)
image(trans3)
as(trans3, "data.frame")

a_df <- sample(c(LETTERS[1:5],NA),10,TRUE)
a_df <- data.frame(X =a_df, Y= sample(a_df))
a_df
trans4 <- as(a_df, "transactions")
trans4
as(trans4, "data.frame")


data(Adult)
class(Adult)

arules::itemFrequency(Adult)
arules::itemFrequencyPlot(Adult) #도수 막대그래프,#type="absolute"
arules::itemFrequencyPlot(Adult,support=0.1,main="item frequency plot support") #지지도 1% 이상의 item에 대한 막대그래프
arules::itemFrequencyPlot(Adult, topN=30,cex.names=0.8 ,main="support top 30 item") #support 상위 30개의 막대그래프
#image(Adult)

rules <- arules::apriori(Adult, parameter = list(supp=0.5, conf=0.9, target = "rules"))
rules1 <- arules::apriori(Adult, parameter = list(supp=0.4)) # apriori default supp = 0.1, conf = 0.8
summary(rules)
inspect(rules)

summary(rules1)
inspect(rules1)

rules1

rules.sub<-arules::subset(rules1, subset = rhs %pin% "sex" & lift >= 1.3)
inspect(sort(rules.sub))


load("C:/Users/korea/Downloads/titanic.raw.rdata")inspect(sort(arules::subset(rules1, subset = lhs %pin% "sex" & lift >= 1.3)))
dim(titanic.raw)


rules = apriori(titanic.raw, parameter = list(minlen=2 ,supp=0.005, conf=0.8), 
                appearance = list(rhs=c("Survived=Yes","Survived=No")))

inspect(sort(rules, by = "lift"))
x<-sort(rules, by = "lift")
s.x<-as.matrix(is.subset(x,x))
s.x[lower.tri(s.x,diag=T)]=NA
redundant <- colSums(s.x, na.rm=T) >=1
which(redundant)
X<-x[!redundant]
inspect(X)
plot(X)
plot(X, method= "graph", control=list(type="rules"))
plot(X, method= "graph", control=list(type="items"))

plot(X,method="graph" )
#?arulesViz::plot
plot(X,method="graph")
plot(X,method="paracoord")

# 자세한 것은 help


data(Groceries)
rules <- apriori(Groceries, parameter=list(support=0.001, confidence=0.8))
rules




# itemset
itemsets <- eclat(Groceries, parameter = list(support = 0.02, minlen=2))
inspect(itemsets)
plot(itemsets)
plot(itemsets, method="graph")
plot(itemsets, method="paracoord", alpha=.5, reorder=TRUE)





## Scatterplot
## -----------
plot(rules)

## Scatterplot with custom colors
library(colorspace) # for sequential_hcl
plot(rules, control = list(col=sequential_hcl(100)))
plot(rules, col=sequential_hcl(100))
plot(rules, col=grey.colors(50, alpha =.8))

## See all control options using verbose
plot(rules, verbose = F)

## Interactive plot (selected rules are returned)
## Not run: 
sel <- plot(rules, engine = "interactive")
## End(Not run)

## Create a html widget for interactive visualization
## Not run: 
plot(rules, engine = "htmlwidget")
## End(Not run)

## Two-key plot (is a scatterplot with shading = "order")
plot(rules, method = "two-key plot")
rules

## Matrix shading
## --------------

## The following techniques work better with fewer rules
subrules <- subset(rules, lift>5)
subrules

## 2D matrix with shading
plot(subrules, method="matrix")

## 3D matrix
plot(subrules, method="matrix", engine = "3d")

## Matrix with two measures
plot(subrules, method="matrix", shading=c("lift", "confidence"))

## Interactive matrix plot (default interactive and as a html widget)
## Not run: 
plot(subrules, method="matrix", engine="interactive")
plot(subrules, method="matrix", engine="htmlwidget")
## End(Not run)

## Grouped matrix plot
## -------------------

plot(rules, method="grouped matrix")
plot(rules, method="grouped matrix", 
     col = grey.colors(10), 
     gp_labels = gpar(col = "blue", cex=1, fontface="italic"))

## Interactive grouped matrix plot
## Not run: 
sel <- plot(rules, method="grouped", engine = "interactive")
## End(Not run)

## Graphs
## ------

## Graphs only work well with very few rules
subrules2 <- sample(subrules, 25)

plot(subrules2, method="graph")

## Custom colors
plot(subrules2, method="graph", 
     nodeCol = grey.colors(10), edgeCol = grey(.7), alpha = 1)

## igraph layout generators can be used (see ? igraph::layout_)
plot(subrules2, method="graph", layout=igraph::in_circle())
plot(subrules2, method="graph", 
     layout=igraph::with_graphopt(spring.const=5, mass=50))

## Graph rendering using Graphviz
## Not run: 
plot(subrules2, method="graph", engine="graphviz")
## End(Not run)

## Default interactive plot (using igraph's tkplot)
## Not run: 
plot(subrules2, method="graph", engine = "interactive")
## End(Not run)

## Interactive graph as a html widget (using igraph layout)
## Not run: 
plot(subrules2, method="graph", engine="htmlwidget")
plot(subrules2, method="graph", engine="htmlwidget", 
     igraphLayout = "layout_in_circle")

## End(Not run)

## Parallel coordinates plot
## -------------------------

plot(subrules2, method="paracoord")
plot(subrules2, method="paracoord", reorder=TRUE)

## Doubledecker plot 
## -----------------

## Note: only works for a single rule
oneRule <- sample(rules, 1)
inspect(oneRule)
plot(oneRule, method="doubledecker", data = Groceries)

## Itemsets
## --------

itemsets <- eclat(Groceries, parameter = list(support = 0.02, minlen=2))
inspect(itemsets)
plot(itemsets)
plot(itemsets, method="graph")
plot(itemsets, method="paracoord", alpha=.5, reorder=TRUE)

## Add more quality measures to use for the scatterplot
## ----------------------------------------------------

quality(itemsets) <- interestMeasure(itemsets, trans=Groceries)
head(quality(itemsets))
plot(itemsets, measure=c("support", "allConfidence"), shading="lift")

## Save HTML widget as web page
## ----------------------------
## Not run: 
p <- plot(rules, engine = "html")
htmlwidgets::saveWidget(p, "arules.html", selfcontained = FALSE)
browseURL("arules.html")
## End(Not run)
# Note: selfcontained seems to make the browser slow.