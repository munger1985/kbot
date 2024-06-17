import oci
config = oci.config.from_file()

object_storage_client = oci.object_storage.ObjectStorageClient(config)
namespace='sehubjapacprod'
bucket='testo'
response = object_storage_client.list_objects(namespace, bucket,
                                              prefix='',
                                              fields='size,timeCreated,timeModified,storageTier',
                                              retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)

sdsd=response.data
asd=3


adasd= 'dd.wav'
df=adasd.endswith('.wav')
print(df)
filename = 'asdsad/lplsd/sdsd'
filename=  filename.replace('/','-')
print(filename)
