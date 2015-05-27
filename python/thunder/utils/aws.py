"""
Utilities for interacting with Amazon Web Services
"""


class AWSCredentials(object):
    __slots__ = ('awsAccessKeyId', 'awsSecretAccessKey')

    def __init__(self, awsAccessKeyId=None, awsSecretAccessKey=None):
        self.awsAccessKeyId = awsAccessKeyId if awsAccessKeyId else None
        self.awsSecretAccessKey = awsSecretAccessKey if awsSecretAccessKey else None

    def __repr__(self):
        def obfuscate(s):
            return "None" if s is None else "<%d-char string>" % len(s)
        return "AWSCredentials(accessKeyId: %s, secretAccessKey: %s)" % \
               (obfuscate(self.awsAccessKeyId), obfuscate(self.awsSecretAccessKey))

    def setOnContext(self, sparkContext):
        sparkContext._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", self.awsAccessKeyId)
        sparkContext._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", self.awsSecretAccessKey)

    @classmethod
    def fromContext(cls, sparkContext):
        if sparkContext:
            awsAccessKeyId = sparkContext._jsc.hadoopConfiguration().get("fs.s3n.awsAccessKeyId", "")
            awsSecretAccessKey = sparkContext._jsc.hadoopConfiguration().get("fs.s3n.awsSecretAccessKey", "")
            return AWSCredentials(awsAccessKeyId, awsSecretAccessKey)
        else:
            return AWSCredentials()

    @property
    def credentials(self):
        if self.awsAccessKeyId and self.awsSecretAccessKey:
            return self.awsAccessKeyId, self.awsSecretAccessKey
        else:
            return None, None

    @property
    def credentialsAsDict(self):
        access, secret = self.credentials
        return {"aws_access_key_id": access, "aws_secret_access_key": secret}

