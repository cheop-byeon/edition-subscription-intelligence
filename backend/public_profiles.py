"""Artificial UI examples. These are not rows from the downloaded benchmark."""
def make_profiles(artifact):
    from external_engine import predict_export
    profiles=[]
    for i in range(20):
        days7=1+i%7;days28=min(28,days7*3+i%4);songs=2+i*4;seconds=songs*(80+i*5)
        values={'num_unq':songs,'num_100':int(songs*.65),'num_25':int(songs*.15),'num_50':int(songs*.1),'num_75':int(songs*.05),'num_985':int(songs*.05),'total_secs':seconds,'avg_num_unq':songs*.85,'smooth_avg_num_unq':songs*.9,'max_num_unq':songs*1.3,'min_num_unq':max(1,int(songs*.2)),'days_since_first_visit':i*20,'visit_frequency_last7days':days7,'visit_frequency_last14days':min(14,days7*2),'visit_frequency_last28days':days28,'volume_total_secs_last7days':seconds*days7,'volume_total_secs_last14days':seconds*min(14,days7*2),'volume_total_secs_last28days':seconds*days28,'volume_num_100_last7days':int(songs*.65)*days7,'volume_num_100_last28days':int(songs*.65)*days28}
        profiles.append({'id':f'example-{i+1:03d}','date':None,'label':None,'kind':'artificial_input','prediction':predict_export(values,artifact['model']),'values':values})
    return profiles
