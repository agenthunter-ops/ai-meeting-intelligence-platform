import grpc
from concurrent import futures
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock whisper_cpp for now - in production you'd install the real package
class MockWhisperCpp:
    def transcribe(self, file_path):
        # Mock transcription for testing
        return [
            type('Segment', (), {'start': 0.0, 'end': 2.0, 'text': 'Hello world'})(),
            type('Segment', (), {'start': 2.0, 'end': 4.0, 'text': 'This is a test'})()
        ]

whisper_cpp = MockWhisperCpp()

# Mock protobuf classes for now
class MockProto:
    class TranscribeRequest:
        def __init__(self, file_path):
            self.file_path = file_path
    
    class Segment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

meeting_pb2 = MockProto()
meeting_pb2_grpc = type('MockGrpc', (), {})()

class Transcriber:
    def Transcribe(self, request, context):
        for seg in whisper_cpp.transcribe(request.file_path):
            yield meeting_pb2.Segment(start=seg.start, end=seg.end, text=seg.text)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor())
    # Mock the add_MeetingServicer_to_server function
    def mock_add_servicer(servicer, server):
        pass
    meeting_pb2_grpc.add_MeetingServicer_to_server = mock_add_servicer
    meeting_pb2_grpc.add_MeetingServicer_to_server(Transcriber(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Whisper service started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
