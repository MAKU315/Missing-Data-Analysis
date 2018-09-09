library("Hmisc")

setwd("C:/Users/korea/Desktop/dataset_kor")

# ML 결과 파일
ML_result_mat<-as.matrix(read.csv("ML_result_mat.csv",sep=','))
## 예비 용
ML_result_mat <- ML_result_mat[order(ML_result_mat[,1],ML_result_mat[,2]),]
##
ML_result_mat1 <- ML_result_mat[,1:3]

# 원본 
test_kor<-as.matrix(read.csv("test_kor.csv",sep=','))
test_kor[test_kor==''] <- NA

test_col <- c("주야",	"요일",	"사망자수",	"사상자수",	"중상자수",	"경상자수",	"부상신고자수",	"발생지시도",	
              "발생지시군구",	"사고유형_대분류",	"사고유형_중분류",	"법규위반",	"도로형태_대분류",	"도로형태",	"당사자종별_1당_대분류",	"당사자종별_2당_대분류")

cat_list1 <- c("주야", "요일", "사고유형_대분류","사고유형_중분류", "법규위반", "도로형태_대분류", "도로형태",
               "당사자종별_1당_대분류", "당사자종별_2당_대분류","발생지시도")
num_list1 <- c( "사망자수","중상자수","경상자수","부상신고자수")
num_list2 <- c( "사망자수","중상자수","경상자수","부상신고자수","사상자수")

# 테스트 셋 서치 ## for 문 for(i in 
colnames(test_kor) <- test_col 

last.num <-0
test_kor1 <- test_kor
for(i in 1:dim(test_kor)[1]){  # dim(test_kor)[1]
  test_obs <- test_kor[i,] 
  na.list <- which(is.na(test_obs))
  # 빈 변수 채우기
  for(n.idx in 1:length(na.list))
  {
    #n.idx <-3
    # 수치 결과 채우기 / 사상자 제외
    if(names(test_obs)[na.list][n.idx] %in% num_list1 )
    {
      result <- as.numeric(ML_result_mat[last.num + n.idx , 3:10 * (as.numeric(ML_result_mat[last.num+n.idx,11:18]) != 0) ])
      weight <- as.numeric(ML_result_mat[last.num + n.idx , 11:18 * (as.numeric(ML_result_mat[last.num+n.idx,11:18]) != 0) ])
      weight <- weight/sum(weight)
      
      test_kor1[i ,match(ML_result_mat[last.num + n.idx,2],LETTERS)] <- sum(result*weight)
      ML_result_mat1[last.num + n.idx,3] <- sum(result*weight)
    }
    
    if(names(test_obs)[na.list][n.idx] %in% "사상자수"  )
    {
      result <- as.numeric(ML_result_mat[last.num + n.idx , 3])
      test_kor1[i ,match(ML_result_mat[last.num + n.idx,2],LETTERS)] <- result
      ML_result_mat1[last.num + n.idx,3] <-  result
      
    }    
    # 범주 결과 채우기
    if(names(test_obs)[na.list][n.idx] %nin% num_list2 )
    {

      result <- as.character(ML_result_mat[last.num + n.idx , 3:10 * (ML_result_mat[last.num+n.idx,11:18] != 0) ])
      weight <- as.numeric(ML_result_mat[last.num + n.idx , 11:20 * (ML_result_mat[last.num+n.idx,11:20] != 0) ])
      weight <- weight/sum(weight)
      
      
      uni.rslt <- unique(result)
      wt.vote <- matrix(0, nrow =  1,ncol=length(uni.rslt))
      colnames(wt.vote) <- uni.rslt 
      
      for(uni.idx in 1:length(uni.rslt))
      {
        wt.vote[1,uni.idx] <- sum(weight[result == uni.rslt[uni.idx]])
        
      }
      #View(test_kor1)
      test_kor1[i, match(ML_result_mat[last.num + n.idx,2],LETTERS)] <- colnames(wt.vote)[which.max(wt.vote)]
      ML_result_mat1[last.num + n.idx,3] <-  colnames(wt.vote)[which.max(wt.vote)]
    }
  }
  
  last.num <- last.num + sum(is.na(test_obs))
}

write.csv(test_kor1,"ML_result_input.csv",row.names = F)
write.csv(ML_result_mat1,"ML_result_ans.csv",row.names = F)
