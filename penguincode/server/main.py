"""PenguinCode gRPC Server - Main entry point.

Usage:
    python -m penguincode.server [--host HOST] [--port PORT]
"""

import asyncio
import logging
import signal
from concurrent import futures
from typing import Optional

import grpc

from penguincode.config.settings import Settings, load_settings
from penguincode.proto import (
    add_AuthServiceServicer_to_server,
    add_ChatServiceServicer_to_server,
    add_ToolCallbackServiceServicer_to_server,
    add_HealthServiceServicer_to_server,
)

from .services.auth import AuthServiceImpl
from .services.chat import ChatServiceImpl
from .services.tools import ToolCallbackServiceImpl
from .services.health import HealthServiceImpl
from .interceptors import JWTValidationInterceptor

logger = logging.getLogger(__name__)


class PenguinCodeServer:
    """Main gRPC server for PenguinCode.

    Manages the lifecycle of the gRPC server and all services.
    """

    def __init__(
        self,
        settings: Settings,
        host: str = "localhost",
        port: int = 50051,
    ):
        self.settings = settings
        self.host = host
        self.port = port
        self.server: Optional[grpc.aio.Server] = None

        # Service implementations
        self.auth_service: Optional[AuthServiceImpl] = None
        self.chat_service: Optional[ChatServiceImpl] = None
        self.tool_service: Optional[ToolCallbackServiceImpl] = None
        self.health_service: Optional[HealthServiceImpl] = None

    async def start(self) -> None:
        """Start the gRPC server."""
        # Create interceptors
        interceptors = []
        if self.settings.auth.enabled:
            jwt_interceptor = JWTValidationInterceptor(
                jwt_secret=self.settings.auth.jwt_secret,
                excluded_methods=[
                    "/penguincode.AuthService/Authenticate",
                    "/penguincode.HealthService/Check",
                ],
            )
            interceptors.append(jwt_interceptor)

        # Create async server
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            interceptors=interceptors,
        )

        # Initialize services
        self.auth_service = AuthServiceImpl(self.settings.auth)
        self.chat_service = ChatServiceImpl(self.settings)
        self.tool_service = ToolCallbackServiceImpl()
        self.health_service = HealthServiceImpl(self.settings)

        # Register services
        add_AuthServiceServicer_to_server(self.auth_service, self.server)
        add_ChatServiceServicer_to_server(self.chat_service, self.server)
        add_ToolCallbackServiceServicer_to_server(self.tool_service, self.server)
        add_HealthServiceServicer_to_server(self.health_service, self.server)

        # Configure TLS if enabled
        if self.settings.server.tls_enabled:
            with open(self.settings.server.tls_cert_path, "rb") as f:
                cert = f.read()
            with open(self.settings.server.tls_key_path, "rb") as f:
                key = f.read()
            credentials = grpc.ssl_server_credentials([(key, cert)])
            self.server.add_secure_port(f"{self.host}:{self.port}", credentials)
            logger.info(f"Server starting with TLS on {self.host}:{self.port}")
        else:
            self.server.add_insecure_port(f"{self.host}:{self.port}")
            logger.info(f"Server starting on {self.host}:{self.port}")

        await self.server.start()
        logger.info("PenguinCode gRPC Server started")

    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server gracefully."""
        if self.server:
            logger.info("Stopping server...")
            await self.server.stop(grace_period)
            logger.info("Server stopped")

    async def wait_for_termination(self) -> None:
        """Wait for the server to be terminated."""
        if self.server:
            await self.server.wait_for_termination()


async def serve(
    config_path: str = "config.yaml",
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """Start the PenguinCode gRPC server.

    Args:
        config_path: Path to configuration file
        host: Override host from config
        port: Override port from config
    """
    # Load settings
    try:
        settings = load_settings(config_path)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        settings = Settings()

    # Use overrides or config values
    server_host = host or settings.server.host
    server_port = port or settings.server.port

    # Create and start server
    server = PenguinCodeServer(settings, server_host, server_port)

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await server.start()
        print(f"PenguinCode Server running on {server_host}:{server_port}")
        print("Press Ctrl+C to stop")

        # Wait for stop signal
        await stop_event.wait()

    finally:
        await server.stop()


def main():
    """CLI entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="PenguinCode gRPC Server")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--host", "-H", default=None, help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=None, help="Port to listen on")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(serve(args.config, args.host, args.port))


if __name__ == "__main__":
    main()
