library(ggseg)
library(ggplot2)


setwd('~/Documents/Doctorado_CIC/Internship/Sylvain/Stability-project/Stability_project_Code/example4Luc 4/')

cbbPalette= c("#4f89e0", "#f5ec6c",'#156605',"#76D7C4", '#4d3d87',   "#f5ec6c",'#D81B99')

df=read.csv('./dka_data_Catell.csv', header = TRUE, stringsAsFactors = FALSE)
colnames(df)[1]='region'

df$hemi =''
df$hemi[seq(2,68,2)] ='right'
df$hemi[seq(1,67,2)] ='left'

colnames(df)[2]='decoding corr'
colnames(df)[4]='YEO'
df$YEO_con= as.numeric(plyr::mapvalues(df$YEO, unique(df$YEO), 1:7))


ggplot(df) +
  geom_brain(atlas = dk, 
             position = position_brain(hemi ~ side),
             aes(fill = `decoding corr`)) + viridis::scale_fill_viridis(option='magma') + 
  theme_void() 

ggsave('~/Documents/Doctorado_CIC/Internship/Sylvain/Stability-project/Stability_project_Code/example4Luc 4/ROI_Importance.pdf', device = "pdf")
